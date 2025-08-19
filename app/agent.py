# app/agent.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .llm import call_llm

app = FastAPI()

@app.post("/api/")
async def analyze(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        if not question:
            return JSONResponse({"error": "No question provided"}, status_code=400)

        result = call_llm(question)
        return JSONResponse({"output": json.dumps([result])})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
