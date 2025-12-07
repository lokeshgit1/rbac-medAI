from fastapi import FastAPI
from auth.routes import router as auth_router
from docs.routes import router as docs_router
from chat.routes import router as chat_router

app=FastAPI()

app.include_router(auth_router, prefix="/auth")
app.include_router(docs_router, prefix="/docs")
app.include_router(chat_router, prefix="/query")

@app.get("/")
def health_check():
    return {"Hello": "World"}





# def main():
#     print("Hello from server!")


# if __name__ == "__main__":
#     main()
