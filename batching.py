import asyncio
from typing import List
from schema import RequestWrapper
# -------------------------------
# Batching Manager
# -------------------------------
class DynamicBatcher:
    def __init__(self,config,model_service):
        self.config = config
        self.model_service = model_service
        self.request_queue: List[RequestWrapper] = []
        self.queue_lock = asyncio.Lock()
        self.running = False
    
    async def add_request(self, request: RequestWrapper):
        async with self.queue_lock:
            self.request_queue.append(request)

    async def start(self):
        if not self.running:
            self.running = True
            asyncio.create_task(self._batching_loop())

    async def _batching_loop(self):
        while True:
            await asyncio.sleep(self.batch_interval)
            await self.process_batch()

    async def process_batch(self):
        async with self.queue_lock:
            if not self.request_queue:
                return

            batch = self.request_queue[:self.max_batch_size]
            self.request_queue = self.request_queue[self.max_batch_size:]

        # prompts = [req.prompt for req in batch]
        outputs = await self._model_generate(batch)

        for req, output in zip(batch, outputs):
            if not req.future.done():
                req.future.set_result(output)

    async def _model_generate(self, batch: List) -> List[str]:
        """
        This is where you plug your real model call.
        For now, we simulate with sleep.
        """
        # await asyncio.sleep(0.1)  # simulate model latency
        result = []
        for req in batch:
            res = self.model_service.generate(req.prompt,req.use_medusa)
            result.append(res)
        return result