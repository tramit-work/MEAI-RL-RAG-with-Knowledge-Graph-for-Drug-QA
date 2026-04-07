import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from rl_rag_system import train_rl, load_kg_from_folder

app = FastAPI(title="MeAI Medical AI")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load KG
print("Loading Knowledge Graph...")
kg_folder = "data-kg"
G = load_kg_from_folder(kg_folder)
print("KG Loaded:", len(G.nodes()), "nodes")


# Request model
class QuestionRequest(BaseModel):
    question: str


# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        request=request
    )


# Ask API
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    try:
        query = req.question.strip()

        if query == "":
            return {"answer": "Please enter a question."}

        print("User question:", query)

        response = train_rl(
            query=query,
            ground_truth="",
            G=G,
            episodes=2
        )

        if not response:
            response = "No answer generated."

        return {"answer": response}

    except Exception as e:
        print("SERVER ERROR:", e)
        return {"answer": "Internal system error."}