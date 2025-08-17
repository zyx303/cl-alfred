import torch
import torch.nn as nn
import numpy as np
import copy
import math
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.nn import functional as F

from utils.data_loader import StreamDataset, MemoryDataset
from methods.er_baseline import ER


class _LoRALayer(nn.Module):
    """LoRA layer implementation for SD-LoRA"""
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        return self.w(x) + self.w_b(self.w_a(x))


class LoRAAdapter(nn.Module):
    """
    LoRA adapter for linear layers in the cl-alfred model
    """
    def __init__(self, module, rank=10, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.original_module = module
        
        if isinstance(module, nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            
            # Initialize LoRA parameters
            self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=module.weight.device) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=module.weight.device))
            
            # Freeze original parameters
            for param in module.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"LoRA only supports Linear layers, got {type(module)}")
    
    def forward(self, x):
        original_output = self.original_module(x)
        if self.rank > 0:
            # Ensure tensors are on the same device
            device = x.device
            lora_A = self.lora_A.to(device)
            lora_B = self.lora_B.to(device)
            
            lora_output = (x @ lora_A.T @ lora_B.T) * (self.alpha / self.rank)
            return original_output + lora_output
        return original_output


class SDLoRA(ER):
    """
    SD-LoRA (Structured Decomposition LoRA) for continual learning in CL-ALFRED
    
    This method applies LoRA (Low-Rank Adaptation) with structured decomposition
    to the seq2seq model for continual learning of embodied AI tasks.
    """
    
    def __init__(self, n_classes, model, **kwargs):
        super().__init__(n_classes, model, **kwargs)
        
        # SD-LoRA specific parameters
        self.rank = kwargs.get("lora_rank", 10)
        self.alpha = kwargs.get("lora_alpha", 1.0)
        self.adaptation_lr = kwargs.get("adaptation_lr", 1e-4)
        self.ortho_reg_weight = kwargs.get("ortho_reg_weight", 0.1)
        
        # Task-specific LoRA parameters storage
        self.task_lora_params = {}
        self.current_task_id = 0
        
        # Identify adaptable layers and apply LoRA
        self.lora_layers = {}
        self._apply_lora_to_model()
        
        # Setup optimizer for LoRA parameters only
        self._setup_lora_optimizer()
        
        print(f"SD-LoRA initialized with rank={self.rank}, alpha={self.alpha}")
    
    def _apply_lora_to_model(self):
        """Apply LoRA adapters to specific layers in the model"""
        # Target Linear layers in the seq2seq model
        
        def apply_lora_recursive(module, prefix=""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Linear):
                    # Skip embedding layers and some output layers
                    if any(skip_name in full_name.lower() for skip_name in ['emb', 'embed']):
                        continue
                    
                    # Apply LoRA to encoder/decoder linear layers
                    if any(target in full_name.lower() for target in ['enc', 'dec', 'attn', 'proj', 'linear']):
                        try:
                            lora_adapter = LoRAAdapter(child, self.rank, self.alpha)
                            self.lora_layers[full_name] = lora_adapter
                            setattr(module, name, lora_adapter)
                            print(f"Applied LoRA to layer: {full_name}")
                        except Exception as e:
                            print(f"Warning: Could not apply LoRA to {full_name}: {e}")
                else:
                    # Recursively apply to child modules
                    apply_lora_recursive(child, full_name)
        
        apply_lora_recursive(self.model)
        print(f"Applied LoRA to {len(self.lora_layers)} layers total")
    
    def _setup_lora_optimizer(self):
        """Setup optimizer specifically for LoRA parameters"""
        lora_params = []
        for lora_layer in self.lora_layers.values():
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        
        if lora_params:
            self.lora_optimizer = torch.optim.Adam(lora_params, lr=self.adaptation_lr)
        else:
            self.lora_optimizer = None
    
    def orthogonal_regularization(self):
        """Compute orthogonal regularization loss for LoRA parameters"""
        ortho_loss = 0.0
        count = 0
        
        for lora_layer in self.lora_layers.values():
            # Regularize LoRA_A to be orthogonal
            A = lora_layer.lora_A
            if A.size(0) > 1:  # Only if rank > 1
                gram_matrix = torch.mm(A, A.t())
                identity = torch.eye(A.size(0), device=A.device)
                ortho_loss += torch.norm(gram_matrix - identity, 'fro') ** 2
                count += 1
        
        return ortho_loss / max(count, 1)
    
    def save_task_lora_params(self, task_id):
        """Save current LoRA parameters for the given task"""
        task_params = {}
        for name, lora_layer in self.lora_layers.items():
            task_params[name] = {
                'lora_A': lora_layer.lora_A.clone().detach(),
                'lora_B': lora_layer.lora_B.clone().detach()
            }
        self.task_lora_params[task_id] = task_params
        print(f"Saved LoRA parameters for task {task_id}")
    
    def load_task_lora_params(self, task_id):
        """Load LoRA parameters for the given task"""
        if task_id in self.task_lora_params:
            task_params = self.task_lora_params[task_id]
            for name, lora_layer in self.lora_layers.items():
                if name in task_params:
                    lora_layer.lora_A.data.copy_(task_params[name]['lora_A'])
                    lora_layer.lora_B.data.copy_(task_params[name]['lora_B'])
            print(f"Loaded LoRA parameters for task {task_id}")
    
    def add_new_class(self, class_name):
        """Handle new task/class by saving current LoRA params and initializing new ones"""
        # Save current task LoRA parameters
        if hasattr(self, 'current_task_id'):
            self.save_task_lora_params(self.current_task_id)
        
        # Call parent method
        super().add_new_class(class_name)
        
        # Initialize new LoRA parameters for the new task
        self.current_task_id = len(self.exposed_classes) - 1
        self._initialize_new_task_lora()
        
        # Update optimizer for new parameters
        self._setup_lora_optimizer()
    
    def _initialize_new_task_lora(self):
        """Initialize LoRA parameters for a new task"""
        for lora_layer in self.lora_layers.values():
            # Reinitialize LoRA parameters
            nn.init.normal_(lora_layer.lora_A, std=0.01)
            nn.init.zeros_(lora_layer.lora_B)
    
    def online_train(self, sample, batch_size, iterations=1, stream_batch_size=1):
        """Training loop with SD-LoRA adaptation"""
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(
                datalist=sample,
                cls_list=self.exposed_classes,
                data_dir=self.data_dir
            )
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

        info = {}
        for i in range(iterations):
            self.model.train()

            data = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                data += stream_data['batch']
            if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                data += memory_data['batch']

            batch = [(self.model.load_task_json(task), swapColor) for task, swapColor in data]
            feat = self.model.featurize(batch)

            out = self.model.forward(feat)
            
            # Compute standard task loss
            self.optimizer.zero_grad()
            if self.lora_optimizer:
                self.lora_optimizer.zero_grad()
            
            loss_dict = self.model.compute_loss(out, batch, feat)
            task_loss = sum(loss_dict.values())
            
            # Add orthogonal regularization for LoRA parameters
            ortho_loss = self.orthogonal_regularization()
            total_loss = task_loss + self.ortho_reg_weight * ortho_loss
            
            if 'cls_loss' not in info:
                info['cls_loss'] = 0
            info['cls_loss'] += task_loss.item()
            
            if 'ortho_loss' not in info:
                info['ortho_loss'] = 0
            info['ortho_loss'] += ortho_loss.item()
            
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            
            # Update parameters
            self.optimizer.step()
            if self.lora_optimizer:
                self.lora_optimizer.step()
            
            self.update_schedule()

        info = {k: v / iterations for k, v in info.items()}
        return info
    
    def report_training(self, sample_num, info):
        """Enhanced reporting including LoRA-specific metrics"""
        super().report_training(sample_num, info)
        
        # Log LoRA-specific metrics
        if 'ortho_loss' in info:
            self.writer.add_scalar('train/ortho_loss', info['ortho_loss'], sample_num)
        
        # Log LoRA parameter norms
        total_lora_norm = 0.0
        for name, lora_layer in self.lora_layers.items():
            a_norm = torch.norm(lora_layer.lora_A).item()
            b_norm = torch.norm(lora_layer.lora_B).item()
            total_lora_norm += a_norm + b_norm
            self.writer.add_scalar(f'lora_norms/{name}_A', a_norm, sample_num)
            self.writer.add_scalar(f'lora_norms/{name}_B', b_norm, sample_num)
        
        self.writer.add_scalar('lora_norms/total', total_lora_norm, sample_num)
    
    def online_after_task(self, cur_iter):
        """Save LoRA parameters after each task"""
        super().online_after_task(cur_iter)
        self.save_task_lora_params(self.current_task_id)
        print(f"Task {cur_iter} completed, LoRA parameters saved")
    
    def evaluation(self, test_list, sample_num, batch_size=32):
        """Evaluation with current LoRA adaptation"""
        eval_dict = {}
        p_valid, _, total_valid_loss, m_valid = self.model.run_pred(
            test_list, batch_size=batch_size, iter=sample_num
        )
        eval_dict['cls_loss'] = float(total_valid_loss)
        eval_dict.update(self.model.compute_metric(p_valid, test_list))
        return eval_dict
