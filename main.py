import os
from typing import List, Optional, Literal, Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# ----- AI Chat endpoint -----

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_title: Optional[str] = Field(default="New Chat")
    model: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    model: str
    usage: Optional[Dict[str, Any]] = None


def get_ai_client_and_model(requested_model: Optional[str] = None):
    """Return an OpenAI-compatible client and model based on environment.

    Supports:
    - GitHub Models (set AI_PROVIDER=github and GITHUB_TOKEN)
    - OpenAI (set AI_PROVIDER=openai and OPENAI_API_KEY)
    - Custom base URL (set AI_BASE_URL + AI_API_KEY)
    """
    from openai import OpenAI

    provider = (os.getenv("AI_PROVIDER") or "github").lower()
    base_url = os.getenv("AI_BASE_URL")
    api_key = None
    model = requested_model or os.getenv("AI_MODEL") or "gpt-4o-mini"

    if provider == "github":
        # GitHub Models use OpenAI SDK with a special base URL
        api_key = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("GITHUB_MODELS_TOKEN")
        base_url = base_url or "https://models.inference.ai.azure.com"
        if not api_key:
            raise HTTPException(status_code=500, detail="GITHUB_TOKEN not configured on server")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server")
        # base_url default None for OpenAI
    else:
        # Custom provider via OpenAI-compatible endpoint
        api_key = os.getenv("AI_API_KEY")
        if not (api_key and base_url):
            raise HTTPException(status_code=500, detail="AI_BASE_URL and AI_API_KEY must be set for custom provider")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Simple chat endpoint that proxies to an OpenAI-compatible API.
    Also persists the conversation to the database as a chat_session document.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    client, model = get_ai_client_and_model(request.model)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[m.model_dump() for m in request.messages],
            temperature=0.7,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI provider error: {str(e)[:200]}")

    reply = completion.choices[0].message.content if completion and completion.choices else ""

    # Persist chat session
    try:
        session_doc = {
            "title": request.session_title or "New Chat",
            "messages": [m.model_dump() for m in request.messages] + [{"role": "assistant", "content": reply}],
        }
        create_document("chatsession", session_doc)
    except Exception:
        # Ignore persistence errors to not block chat
        pass

    usage = None
    try:
        if hasattr(completion, "usage") and completion.usage:
            usage = completion.usage.model_dump() if hasattr(completion.usage, "model_dump") else dict(completion.usage)
    except Exception:
        usage = None

    return ChatResponse(reply=reply, model=model, usage=usage)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
