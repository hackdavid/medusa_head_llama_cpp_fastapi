# app/services/model_service.py

from model_loader import ModelLoader

class ModelService:
    def __init__(self,config):
        self.config = config
        model_loader = ModelLoader(config=config)
        self.model = model_loader.get_model()

    async def generate(self, prompt: list[str],use_medusa: bool=True) -> list[str]:
        if use_medusa:
            return await self._generate_plain(prompt)
        else:
            return await self._generate_plain(prompt)

    async def _generate_plain(self, prompt: list[str]) -> list[str]:
        output = self.model(
            prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            stop=["</s>"],
        )
        generated_text = output["choices"][0]["text"]
        return generated_text

    async def _generate_with_medusa(self, prompt: list[str]) -> list[str]:
        # Placeholder for Medusa speculative decoding (implement later)
        return ''
