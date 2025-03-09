# server.py
import os
from fastapi import FastAPI
from livekit import api
from pydantic import BaseModel

app = FastAPI()

class TokenResponse(BaseModel):
    token: str

@app.get("/getToken", response_model=TokenResponse)
async def get_token():
    token = api.AccessToken(os.getenv('LIVEKIT_API_KEY', 'devkey'), os.getenv('LIVEKIT_API_SECRET', 'secret')) \
        .with_identity("identity") \
        .with_name("my name") \
        .with_grants(api.VideoGrants(
            room_join=True,
            room="my-room",
        ))
    return {"token": token.to_jwt()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3111)

# python server.py