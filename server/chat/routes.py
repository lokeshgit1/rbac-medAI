from fastapi import APIRouter, Depends, Form, HTTPException
from auth.routes import authenticate
from chat.chat_query import answer_query

router = APIRouter()

@router.post("/chat")
async def chat(
    message: str = Form(...),
    user = Depends(authenticate)
):
    try:
        response = await answer_query(message, user["role"])
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
