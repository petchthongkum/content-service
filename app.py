from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
import httpx

load_dotenv()

app = FastAPI()

API_KEY = os.getenv("CONTENT_MODEL_API_KEY")
MAXPLUS_API_URL = "https://api.maxplus-ai.cc/v1/messages"

class ContentRequest(BaseModel):
    prompt: Optional[str] = None
    content: Optional[str] = None
    task_type: str = "generate"

class ContentResponse(BaseModel):
    result: str

async def call_maxplus_api(prompt: str) -> str:
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": "claude-sonnet-5",
        "max_tokens": 2000,
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    }
    
    async with httpx.AsyncClient(verify=False) as client:
        response = await client.post(MAXPLUS_API_URL, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]

@app.post("/api/content")
async def process_content(request: ContentRequest):
    try:
        if request.task_type == "generate":
            result = await call_maxplus_api(request.prompt)
            
        elif request.task_type == "edit":
            prompt = f"แก้ไข content นี้ตามคำแนะนำ:\n\nContent: {request.content}\n\nคำแนะนำ: {request.prompt}"
            result = await call_maxplus_api(prompt)
            
        elif request.task_type == "summarize":
            prompt = f"สรุป content นี้:\n\n{request.content}"
            result = await call_maxplus_api(prompt)
            
        elif request.task_type == "translate":
            prompt = f"แปลเป็น {request.prompt}:\n\n{request.content}"
            result = await call_maxplus_api(prompt)
        else:
            raise HTTPException(status_code=400, detail="Invalid task_type")
        
        return ContentResponse(result=result)
    
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "service": "content"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
