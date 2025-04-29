from llama_cpp import Llama
import torch

import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)  # initialize as identity
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.linear(x))


class CustomMedusaHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, medusa_num_heads=1, medusa_num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        
        self.projections = nn.ModuleList([
            nn.Sequential(
                *[ResBlock(hidden_size) for _ in range(medusa_num_layers)],
                nn.Linear(hidden_size, vocab_size, bias=False)
            )
            for _ in range(medusa_num_heads)
        ])

    def forward(self, hidden_states):
        """
        hidden_states: tensor of shape (batch_size, hidden_size)
        Returns: list of logits from each head
        """
        logits = []
        for proj in self.projections:
            logits.append(proj(hidden_states))  # (batch_size, vocab_size)
        return logits


class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.medusa_head = None

    def load(self):
        # laoding mesusa head 
        print(f'Loading Medusa head : {self.medusa_head_path}')
        self.medusa_head = CustomMedusaHead(
            hidden_size=self.config.hidden_size,     # depends on your base model
            vocab_size=self.config.vocab_size,       # depends on your base model
            medusa_num_heads=self.config.medusa_nums_heads,          # check config
            medusa_num_layers=self.config.medusa_nums_layers          # check config
        )
        # Load the pretrained weights
        state_dict = torch.load(self.config.medusa_head_path)
        self.medusa_head.load_state_dict(state_dict,strict=False)
        print('Medusa head is loaded successfully .....')

        print(f"Loading model from {self.config.base_model_path}...")
        self.model = Llama(
            model_path=self.config.base_model_path,
            n_ctx=self.config.hidden_size,
            use_mlock=True,      # lock into RAM to avoid swapping
            use_mmap=True,       # memory map to load faster
            logits_all=False,    # don't return logits unless necessary
            seed=42,
            verbose=False,
            draft_model=self.medusa_head
        )
        print(f'Base Model is loaded successfully ..........')

        print("Both Model loaded successfully! ........")

    def get_model(self):
        if self.model is None:
            raise ValueError("Model not loaded yet. Call load() first.")
        return self.model
