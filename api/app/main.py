from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import FastAPI, Request

from pydantic  import BaseModel, StringConstraints

from app.services.chat import RagService , Answer

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = RagService()
    yield
    app.state.rag.close()

app = FastAPI(title="Clause AI APIS", lifespan=lifespan)

class AskQuestion(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
        ]

@app.get(path="/")
def health():
    return {"status": "ok"}

@app.post(path="/ask", response_model=Answer)
def ask_query(body: AskQuestion, req: Request):   
     return req.app.state.rag.ask(body.question)



  
