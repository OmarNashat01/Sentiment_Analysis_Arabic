from fastapi import FastAPI, requests

# from typing import Union
# from pydantic import BaseModel

app = FastAPI()


# class Item(BaseModel):
#     description: Union[str, None] = None
#     price: float
#     tax: Union[float, None] = None

@app.get("/")
async def home(sent: str):
    return (sent, 200)



