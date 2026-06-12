from fastapi import FastAPI, Depends, HTTPException, Security
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import torch
import os
import jwt
from supabase import create_client, Client
from dotenv import load_dotenv

# Load backend environment configurations
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "your-supabase-jwt-secret")
ALGORITHM = "HS256"

# Create a connection client to the Supabase Database
supabase_client: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Error initializing Supabase client: {e}")

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validates the incoming Supabase JWT token."""
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=[ALGORITHM],
            audience="authenticated"
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

def ensure_profile_exists(payload: dict):
    """Ensures that a profile row exists in public.profiles. Auto-creates it from JWT payload if missing."""
    user_id = payload.get("sub")
    email = payload.get("email", "")
    if not supabase_client:
        return
    try:
        response = supabase_client.table("profiles").select("role").eq("id", user_id).execute()
        if not response.data:
            # Profile row is missing. Auto-create it using metadata role
            metadata = payload.get("user_metadata", {}) or {}
            role = metadata.get("role", "student")
            if role not in ["student", "admin"]:
                role = "student"
            supabase_client.table("profiles").insert({
                "id": user_id,
                "email": email,
                "role": role
            }).execute()
    except Exception as e:
        print(f"Error auto-syncing user profile: {e}")

def get_user_role(payload: dict = Depends(verify_token)) -> str:
    """Fetches user role from public.profiles table."""
    ensure_profile_exists(payload)
    user_id = payload.get("sub")
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Database client not initialized")
    try:
        response = supabase_client.table("profiles").select("role").eq("id", user_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="User profile not found")
        return response.data[0]["role"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

def require_admin(role: str = Depends(get_user_role)):
    """Dependency to enforce admin role requirement."""
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin permissions required")
    return role

model_path = "./tinyllama-coding-model"

print("Loading tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    raise


from datetime import datetime

from typing import Optional

class Question(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None

class ConversationCreate(BaseModel):
    title: str = "New Chat"


@app.post("/conversations")
def create_conversation(data: ConversationCreate, payload: dict = Depends(verify_token)):
    """Creates a new conversation session for the user."""
    ensure_profile_exists(payload)
    user_id = payload.get("sub")
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Database client not initialized")
    try:
        response = supabase_client.table("conversations").insert({
            "user_id": user_id,
            "title": data.title
        }).execute()
        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to create conversation")
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations")
def get_conversations(search: str = None, payload: dict = Depends(verify_token)):
    """Retrieves all chat conversations for the logged in user, optional search filtering."""
    ensure_profile_exists(payload)
    user_id = payload.get("sub")
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Database client not initialized")
    try:
        query = supabase_client.table("conversations").select("*").eq("user_id", user_id)
        if search:
            query = query.ilike("title", f"%{search}%")
        response = query.order("updated_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{id}")
def delete_conversation(id: str, payload: dict = Depends(verify_token)):
    """Deletes a conversation and cascades to delete all related messages."""
    ensure_profile_exists(payload)
    user_id = payload.get("sub")
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Database client not initialized")
    try:
        # Check ownership
        check = supabase_client.table("conversations").select("id").eq("id", id).eq("user_id", user_id).execute()
        if not check.data:
            raise HTTPException(status_code=404, detail="Conversation not found or unauthorized")
        
        supabase_client.table("conversations").delete().eq("id", id).execute()
        return {"status": "success", "message": "Conversation deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversations/{id}/messages")
def get_messages(id: str, payload: dict = Depends(verify_token)):
    """Fetches all messages for a specific conversation session."""
    ensure_profile_exists(payload)
    user_id = payload.get("sub")
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Database client not initialized")
    try:
        # Check ownership
        check = supabase_client.table("conversations").select("id").eq("id", id).eq("user_id", user_id).execute()
        if not check.data:
            raise HTTPException(status_code=404, detail="Conversation not found or unauthorized")
        
        response = supabase_client.table("messages").select("*").eq("conversation_id", id).order("created_at", desc=False).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_ai(data: Question, payload: dict = Depends(verify_token)):
    """Handles prompt query, logs history to database, and runs LLM generation."""
    ensure_profile_exists(payload)
    user_id = payload.get("sub")
    if not supabase_client:
        raise HTTPException(status_code=500, detail="Database client not initialized")
    
    conversation_id = data.conversation_id

    # 1. Create a new conversation if conversation_id was not provided
    if not conversation_id:
        try:
            # Set the title to the first few characters of prompt
            title = data.prompt[:40] + "..." if len(data.prompt) > 40 else data.prompt
            conv_response = supabase_client.table("conversations").insert({
                "user_id": user_id,
                "title": title
            }).execute()
            if not conv_response.data:
                raise HTTPException(status_code=500, detail="Failed to create new conversation session")
            conversation_id = conv_response.data[0]["id"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")
    else:
        # Verify ownership of existing conversation
        try:
            check = supabase_client.table("conversations").select("id").eq("id", conversation_id).eq("user_id", user_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="Conversation not found or unauthorized")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 2. Save the user's question to messages
    try:
        supabase_client.table("messages").insert({
            "conversation_id": conversation_id,
            "sender": "user",
            "content": data.prompt
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save user message: {str(e)}")

    # 3. Model generation
    text = f"""
You are a professional coding assistant.
Give clear and correct coding answers.

Instruction: {data.prompt}

Answer:
"""
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = result.replace(text, "").strip()

    # 4. Save the AI's response to messages and update conversation timestamp
    try:
        supabase_client.table("messages").insert({
            "conversation_id": conversation_id,
            "sender": "ai",
            "content": answer
        }).execute()

        supabase_client.table("conversations").update({
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", conversation_id).execute()
    except Exception as e:
        print(f"Error saving AI message or updating timestamp: {e}")

    return {
        "response": answer,
        "conversation_id": conversation_id
    }


@app.get("/admin/status")
def admin_status(admin_role: str = Depends(require_admin)):
    """Protected admin endpoint checking roles."""
    return {
        "status": "online",
        "message": "All backend systems are operational.",
        "admin_role": admin_role
    }

