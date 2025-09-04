from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import openai

router = APIRouter()


class MessageRequest(BaseModel):
    session_id: str | None = None
    content: str


@router.post("/message")
def agent_message(req: MessageRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    client = openai.OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are Trailblip Agent. Be concise and helpful."},
                {"role": "user", "content": req.content},
            ],
        )
        return {
            "success": True,
            "data": {
                "session_id": req.session_id,
                "message": completion.choices[0].message.content,
                "tokens": completion.usage.total_tokens if hasattr(completion, "usage") else 0,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


