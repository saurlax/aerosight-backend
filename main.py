import os
import uvicorn
from fastapi import FastAPI
import aerosight.config
from aerosight.routes import router


app = FastAPI(title="AeroSight API")

app.include_router(router)


@app.get("/")
async def hello():
    return {"message": "Hello AeroSight!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host=os.getenv("SERVER_HOST"),
                port=int(os.getenv("SERVER_PORT")), reload=True)
