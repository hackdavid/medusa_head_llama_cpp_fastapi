from pydantic import BaseModel
import asyncio
import uuid
# -------------------------------
# Request and Response Schemas
# -------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    use_medusa : bool = True

class GenerateResponse(BaseModel):
    request_id: str
    output: str

# -------------------------------
# Request Wrapper
# -------------------------------

class RequestWrapper:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.id = str(uuid.uuid4())
        self.future = asyncio.get_event_loop().create_future()

