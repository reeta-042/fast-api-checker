from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InputSchema(BaseModel):
    name: str

@app.get("/")
def read_root():
    return {"message": "API is live!"}

@app.post("/check")
def check_product(data: InputSchema):
    return {"result": f"Received {data.name}"}
