import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LlamaEncoderConfig:
    model_name_or_path: str = "initial_model/llama/"
    device: str = "cuda"
    dtype: Optional[torch.dtype] = None  # e.g., torch.float16
    max_length: int = 512


class LlamaTextEncoder:
    """
    Thin wrapper around transformers LlamaModel for getting last_hidden_state.
    """

    def __init__(self, config: LlamaEncoderConfig):
        from transformers import AutoModel, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)

        self.tokenizer.add_special_tokens({"pad_token": "<<pad>>"})
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "<<eos>>"})
        else:
            self.tokenizer.eos_token = "<<eos>>"
        
        self.model = AutoModel.from_pretrained(
            config.model_name_or_path,
            torch_dtype=config.dtype,
            low_cpu_mem_usage=True,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.device = torch.device(config.device)
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Returns last hidden states as a padded tensor [B, T, H].
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        return out.last_hidden_state  # [B, T, H]
