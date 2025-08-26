import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LlamaEncoderConfig:
    model_name_or_path: str = "meta-llama/Llama-2-7b-hf"
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
        self.model = AutoModel.from_pretrained(
            config.model_name_or_path,
            torch_dtype=config.dtype,
            low_cpu_mem_usage=True,
        )
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
