import os
import re
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.seq2seq_im_mask import Module as Base

class LoRALinear(nn.Module):
    """
    O-LoRA style linear adapter following official implementation:
      - base LoRA: lora_A/lora_B (frozen, accumulates previous tasks, shape depends on r_sum)
      - new LoRA:  loranew_A/loranew_B (trainable for current task, shape depends on r)
      - forward adds both residuals scaled by lora_alpha/rank
    """
    def __init__(self, original_layer: nn.Linear, rank=4, lora_alpha=32, r_sum=0):
        super().__init__()
        assert isinstance(original_layer, nn.Linear)
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = int(rank)  # current task rank
        self.r_sum = int(r_sum)  # accumulated rank from previous tasks
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / max(self.rank, 1)

        # Freeze original parameters
        for p in self.original_layer.parameters():
            p.requires_grad = False

        # Base LoRA (frozen) - dimensions based on r_sum
        if self.r_sum > 0:
            self.lora_A = nn.Linear(self.in_features, self.r_sum, bias=False)
            self.lora_B = nn.Linear(self.r_sum, self.out_features, bias=False)
        else:
            self.lora_A = None
            self.lora_B = None
            
        # New LoRA (trainable) - dimensions based on current rank
        self.loranew_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.loranew_B = nn.Linear(self.rank, self.out_features, bias=False)

        self._reset_parameters()
        self._freeze_base()

        # ensure dtype/device alignment
        dev = self.original_layer.weight.device
        dtype = self.original_layer.weight.dtype
        self.to(device=dev, dtype=dtype)

    def _freeze_base(self):
        if self.lora_A is not None:
            for p in self.lora_A.parameters():
                p.requires_grad = False
        if self.lora_B is not None:
            for p in self.lora_B.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # base starts from zero (kept frozen until merged)
        if self.lora_A is not None:
            nn.init.zeros_(self.lora_A.weight)
        if self.lora_B is not None:
            nn.init.zeros_(self.lora_B.weight)
            
        # new part follows O-LoRA init - kaiming uniform for A, zeros for B
        nn.init.kaiming_uniform_(self.loranew_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.loranew_B.weight)

    def reset_loranew(self):
        """Reinitialize loranew_* (for a fresh task)."""
        nn.init.kaiming_uniform_(self.loranew_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.loranew_B.weight)

    def merge_new_into_base(self):
        """Accumulate current loranew into base lora and reset loranew.
        This dynamically expands the base LoRA dimensions to accommodate new tasks.
        """
        with torch.no_grad():
            current_r_sum = self.r_sum
            new_r_sum = current_r_sum + self.rank
            
            # Create new expanded base LoRA layers
            new_lora_A = nn.Linear(self.in_features, new_r_sum, bias=False)
            new_lora_B = nn.Linear(new_r_sum, self.out_features, bias=False)
            
            # Move to same device/dtype
            new_lora_A = new_lora_A.to(device=self.loranew_A.weight.device, 
                                       dtype=self.loranew_A.weight.dtype)
            new_lora_B = new_lora_B.to(device=self.loranew_A.weight.device, 
                                       dtype=self.loranew_A.weight.dtype)
            
            # Initialize new layers to zero
            nn.init.zeros_(new_lora_A.weight)
            nn.init.zeros_(new_lora_B.weight)
            
            # Copy existing base LoRA weights if they exist
            if self.lora_A is not None and current_r_sum > 0:
                new_lora_A.weight[:current_r_sum, :].copy_(self.lora_A.weight)
                new_lora_B.weight[:, :current_r_sum].copy_(self.lora_B.weight)
            
            # Add new LoRA weights to the expanded dimensions
            new_lora_A.weight[current_r_sum:new_r_sum, :].copy_(self.loranew_A.weight)
            new_lora_B.weight[:, current_r_sum:new_r_sum].copy_(self.loranew_B.weight)
            
            # Replace base LoRA with expanded version
            self.lora_A = new_lora_A
            self.lora_B = new_lora_B
            self.r_sum = new_r_sum
            
            # Freeze the new base LoRA
            for p in self.lora_A.parameters():
                p.requires_grad = False
            for p in self.lora_B.parameters():
                p.requires_grad = False
            
            # Reset new LoRA for next task
            nn.init.kaiming_uniform_(self.loranew_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.loranew_B.weight)

    def forward(self, x):
        out = self.original_layer(x)
        
        # base residual (frozen) - only compute if base LoRA exists
        if self.lora_A is not None and self.r_sum > 0:
            base_res = self.lora_B(self.lora_A(x))
            out = out + self.scaling * base_res
        
        # new residual (trainable)
        new_res = self.loranew_B(self.loranew_A(x))
        out = out + self.scaling * new_res
        
        return out




def replace_linear_with_lora(module, rank=4, lora_alpha=1.0, r_sum=0):
    """Recursively replace nn.Linear layers with O-LoRA LoRALinear wrappers.
    Guard against re-entering LoRALinear to avoid double wrapping.
    
    Args:
        module: The module to modify
        rank: Current task LoRA rank
        lora_alpha: LoRA scaling factor  
        r_sum: Accumulated rank from previous tasks
    """
    for name, child in module.named_children():
        # skip if already a LoRALinear wrapper
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, lora_alpha=lora_alpha, r_sum=r_sum))
        else:
            replace_linear_with_lora(child, rank=rank, lora_alpha=lora_alpha, r_sum=r_sum)


def replace_attention_with_lora(module, rank=4, lora_alpha=1.0, r_sum=0):
    """Optionally replace known attention projections with LoRA versions (kept for parity).
    
    Args:
        module: The module to modify
        rank: Current task LoRA rank
        lora_alpha: LoRA scaling factor
        r_sum: Accumulated rank from previous tasks
    """
    for name, child in module.named_children():
        # skip if already a LoRALinear wrapper
        if isinstance(child, LoRALinear):
            continue
        if hasattr(child, 'fc_key') and hasattr(child, 'fc_query'):
            if isinstance(child.fc_key, nn.Linear):
                child.fc_key = LoRALinear(child.fc_key, rank=rank, lora_alpha=lora_alpha, r_sum=r_sum)
            if isinstance(child.fc_query, nn.Linear):
                child.fc_query = LoRALinear(child.fc_query, rank=rank, lora_alpha=lora_alpha, r_sum=r_sum)
        elif hasattr(child, 'scorer') and isinstance(child.scorer, nn.Linear):
            child.scorer = LoRALinear(child.scorer, rank=rank, lora_alpha=lora_alpha, r_sum=r_sum)
        else:
            replace_attention_with_lora(child, rank=rank, lora_alpha=lora_alpha, r_sum=r_sum)


class Module(Base):
    """Seq2Seq agent with O-LoRA adapters on every Linear layer."""

    def __init__(self, args, vocab):
        super().__init__(args, vocab)
        self.lora_rank = int(getattr(args, 'lora_rank', 8))
        self.lora_alpha = float(getattr(args, 'lora_alpha', 32))
        self.r_sum = int(getattr(args, 'r_sum', 0))  # accumulated rank from previous tasks
        
        # wrap linear layers
        replace_linear_with_lora(self, rank=self.lora_rank, lora_alpha=self.lora_alpha, r_sum=self.r_sum)
        # optionally: attention projections (disabled by default)
        # replace_attention_with_lora(self, rank=self.lora_rank, lora_alpha=self.lora_alpha, r_sum=self.r_sum)

    # -------- helper APIs for O-LoRA method --------
    def reset_loranew(self):
        """Reset all new LoRA parameters for a fresh task."""
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.reset_loranew()

    def finalize_lora_task(self):
        """Merge new LoRA into base and update r_sum."""
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.merge_new_into_base()
        # Update global r_sum after merging
        self.r_sum += self.lora_rank
        
    def get_current_r_sum(self):
        """Get the current accumulated rank."""
        return self.r_sum
        
    def set_r_sum(self, r_sum):
        """Set the accumulated rank (used when loading from checkpoint)."""
        self.r_sum = r_sum
        for m in self.modules():
            if isinstance(m, LoRALinear):
                m.r_sum = r_sum

    @classmethod
    def load(cls, fsave):
        """Lightweight loader that restores model/optimizer state."""
        
        save = torch.load(fsave, map_location='cpu')

        r_sum = 0
        if 'r_sum' in save:
            r_sum = int(save['r_sum'])
        args = save['args']
        if not hasattr(args, 'r_sum'):
            args.r_sum = r_sum
        else:
            args.r_sum = r_sum

            
        model = cls(save['args'], save['vocab'])
        # Restore r_sum if present in save
        if hasattr(save.get('args', {}), 'r_sum'):
            model.set_r_sum(save['args'].r_sum)
        state = save.get('model', save.get('model_state_dict', {}))
        
        # Handle key mapping for LoRA-wrapped model
        if any('original_layer' in key for key in state.keys()):
            # State dict is already LoRA-wrapped
            model.load_state_dict(state, strict=True)
        else:
            # State dict is from vanilla model, need to map keys
            new_state = {}
            for key, value in state.items():
                # Check if this layer was wrapped with LoRA
                layer_found = False
                for name, module in model.named_modules():
                    if isinstance(module, LoRALinear) and key.startswith(name + '.'):
                        # Map to original_layer
                        new_key = key.replace(name + '.', name + '.original_layer.')
                        new_state[new_key] = value
                        layer_found = True
                        break
                if not layer_found:
                    new_state[key] = value
            model.load_state_dict(new_state, strict=False)

        # optimizer (if present) — best-effort
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if 'optim' in save:
            try:
                optimizer.load_state_dict(save['optim'])
            except Exception as e:
                print(f"[LoRA load] Optimizer state load failed: {e}")
        return model, optimizer
    
    