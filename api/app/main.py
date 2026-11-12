from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Request
from pydantic import BaseModel, StringConstraints

from app.services.chat import RAGService, Answer

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RAGService()
    yield
    app.state.rag.close()

app = FastAPI(title="Clause AI APIs", lifespan=lifespan)

class ASKQuestion(BaseModel):
    question: Annotated[
      str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask_query(body: ASKQuestion, req: Request):
    return req.app.state.rag.ask(body.question)



  
