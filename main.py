from fastapi import FastAPI, HTTPException
import asyncio
from batching import DynamicBatcher
from schema import GenerateResponse,GenerateRequest,RequestWrapper
import torch
from pydantic import BaseModel
from model import ModelService
'''
python3 convert-hf-to-gguf.py --outtype f16 --outfile ./vicuna_7b_v1.gguf lmsys/vicuna-7b-v1.3


'''

# -------------------------------
# Configuration
# -------------------------------

class Config(BaseModel):
    batch_interval: float = 0.05
    max_batch_size: int = 32
    request_timeout: int = 10
    base_model_device: str = 'cpu'
    medusa_head_device: str = 'cpu'
    medusa_nums_heads: int = 2
    medusa_nums_layers: int = 1
    base_model_path: str = ''
    medusa_head_path: str = ''
    hidden_size: int = 4096
    vocab_size: int = 32000
    d_type: torch.dtype = torch.bfloat16
# -------------------------------
# FastAPI App Initialization
# -------------------------------

app = FastAPI()

# Get the models loaded first
model_service = ModelService(config=Config)
batcher = DynamicBatcher(config=Config,model_service=model_service)

@app.on_event("startup")
async def startup_event():
    await batcher.start()

# -------------------------------
# API Endpoints
# -------------------------------

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(req: GenerateRequest):
    wrapper = RequestWrapper(req.prompt)
    await batcher.add_request(wrapper)

    try:
        output = await asyncio.wait_for(wrapper.future)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timeout, please try again.")

    return GenerateResponse(
        request_id=wrapper.id,
        output=output
    )