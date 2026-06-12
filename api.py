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

def get_user_role(payload: dict = Depends(verify_token)) -> str:
    """Fetches user role from public.profiles table."""
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


class Question(BaseModel):
    prompt: str


@app.post("/ask")
def ask_ai(data: Question, payload: dict = Depends(verify_token)):

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

    return {"response": answer}


@app.get("/admin/status")
def admin_status(admin_role: str = Depends(require_admin)):
    """Protected admin endpoint checking roles."""
    return {
        "status": "online",
        "message": "All backend systems are operational.",
        "admin_role": admin_role
    }

