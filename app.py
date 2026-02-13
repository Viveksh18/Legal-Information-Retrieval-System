from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from retriever import ask_rag
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(req: QueryRequest):
    return {"answer": ask_rag(req.query)}

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
