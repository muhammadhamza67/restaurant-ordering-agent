from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from restaurant_agent import run_agent_turn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-session conversation history, same pattern as your other agent servers
conversations = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


@app.post("/chat")
def chat(req: ChatRequest):
    history = conversations.get(req.session_id, [])
    reply, updated_history = run_agent_turn(req.session_id, req.message, history)
    conversations[req.session_id] = updated_history
    return {"answer": reply}


@app.get("/")
def health_check():
    return {"status": "Restaurant ordering agent server is running"}