from fastapi import FastAPI
from .routes import router as chat_router

app = FastAPI(
    title="Translated Chat Bot",
    version="1.0.0",
    description="A Chat Bot that uses Langchain and OpenAI."
)

# Rotaları ekleyelim
app.include_router(chat_router, prefix="/chain")

@app.get("/")
async def root():
    return {"message": "Welcome to the Chat Bot API!"}
