import os
import torch
import numpy as np
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq_im_mask import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
from models.nn import vnn
import models.nn.vnn as vnn

import constants
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']


class MultiTaskLoRALayer(nn.Module):
    """
    Multi-task LoRA layer that supports multiple tasks with independent adapters
    """
    def __init__(self, in_features, out_features, rank=4, num_tasks=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.num_tasks = num_tasks
        self.current_task = 0
        
        # Create LoRA layers for each task
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.scales = nn.ParameterDict()
        
        for i in range(num_tasks):
            task_name = f"task{i}"
            self.lora_A[task_name] = nn.Linear(in_features, rank, bias=False)
            self.lora_B[task_name] = nn.Linear(rank, out_features, bias=False)
            self.scales[task_name] = nn.Parameter(torch.tensor([0.8]))
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A[task_name].weight, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B[task_name].weight)
    
    def add_task(self, task_id):
        """Add a new task with its own LoRA layers"""
        task_name = f"task{task_id}"
        if task_name not in self.lora_A:
            self.lora_A[task_name] = nn.Linear(self.in_features, self.rank, bias=False)
            self.lora_B[task_name] = nn.Linear(self.rank, self.out_features, bias=False)
            self.scales[task_name] = nn.Parameter(torch.tensor([0.8]))
            
            # Initialize weights
            nn.init.kaiming_uniform_(self.lora_A[task_name].weight, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B[task_name].weight)
            
            # Move to same device as existing parameters
            if len(self.lora_A) > 1:
                device = next(iter(self.lora_A.values())).weight.device
                self.lora_A[task_name] = self.lora_A[task_name].to(device)
                self.lora_B[task_name] = self.lora_B[task_name].to(device)
                self.scales[task_name] = self.scales[task_name].to(device)
            
            self.num_tasks = max(self.num_tasks, task_id + 1)
    
    def set_task(self, task_id):
        """Set the current active task"""
        self.current_task = task_id
        task_name = f"task{task_id}"
        if task_name not in self.lora_A:
            self.add_task(task_id)
    
    def forward(self, x):
        """Forward pass using current task's LoRA layers"""
        task_name = f"task{self.current_task}"
        if task_name not in self.lora_A:
            return torch.zeros_like(x[..., :self.out_features])
        
        result = self.lora_A[task_name](x)
        result = self.lora_B[task_name](result)
        return result * self.scales[task_name]


class LoRALinear(nn.Module):
    """
    Linear layer with multi-task LoRA adaptation
    """
    def __init__(self, original_layer, rank=4, num_tasks=1):
        super().__init__()
        self.original_layer = original_layer
        self.lora = MultiTaskLoRALayer(
            original_layer.in_features, 
            original_layer.out_features, 
            rank=rank, 
            num_tasks=num_tasks
        )
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.original_layer(x) + self.lora(x)
    
    def set_task(self, task_id):
        self.lora.set_task(task_id)
    
    def add_task(self, task_id):
        self.lora.add_task(task_id)


def replace_linear_with_lora(module, rank=4, num_tasks=1, target_modules=None):
    """
    Recursively replace Linear layers with LoRA versions
    """
    if target_modules is None:
        target_modules = ['Linear']
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace with LoRA version
            lora_linear = LoRALinear(child, rank=rank, num_tasks=num_tasks)
            setattr(module, name, lora_linear)
        else:
            # Recursively apply to child modules
            replace_linear_with_lora(child, rank, num_tasks, target_modules)


def replace_attention_with_lora(module, rank=4, num_tasks=1):
    """
    Replace attention q, v projections with LoRA versions
    """
    for name, child in module.named_children():
        if hasattr(child, 'fc_key') and hasattr(child, 'fc_query'):
            # This looks like a scaled dot attention module - replace q,v (key,query)
            if isinstance(child.fc_key, nn.Linear):
                child.fc_key = LoRALinear(child.fc_key, rank=rank, num_tasks=num_tasks)
            if isinstance(child.fc_query, nn.Linear):
                child.fc_query = LoRALinear(child.fc_query, rank=rank, num_tasks=num_tasks)
        
        elif hasattr(child, 'scorer') and isinstance(child.scorer, nn.Linear):
            # This looks like a self attention module
            child.scorer = LoRALinear(child.scorer, rank=rank, num_tasks=num_tasks)
        
        else:
            # Recursively apply to child modules
            replace_attention_with_lora(child, rank, num_tasks)


class Module(Base):
    """
    Seq2Seq agent with multi-task LoRA adaptation
    """
    
    def __init__(self, args, vocab):
        '''
        Initialize Seq2Seq agent with LoRA adaptation
        '''
        super().__init__(args, vocab)
        
        # LoRA configuration
        self.lora_rank = getattr(args, 'lora_rank', 10)
        self.num_tasks = getattr(args, 'num_tasks', 1)
        self.current_task = 0
        
        # Apply LoRA to all linear layers and attention modules
        self._apply_lora_to_model()
        
        # Keep track of LoRA modules for task switching
        self.lora_modules = []
        self._collect_lora_modules()
    
    def _apply_lora_to_model(self):
        """Apply LoRA to all linear layers and attention q,v projections"""
        # Replace linear layers with LoRA versions
        replace_linear_with_lora(self, rank=self.lora_rank, num_tasks=self.num_tasks)
        
        # Replace attention q,v projections with LoRA versions
        # replace_attention_with_lora(self, rank=self.lora_rank, num_tasks=self.num_tasks)
    
    def _collect_lora_modules(self):
        """Collect all LoRA modules for easy task switching"""
        self.lora_modules = []
        for module in self.modules():
            if isinstance(module, (LoRALinear, MultiTaskLoRALayer)):
                self.lora_modules.append(module)
    
    def add_task(self, task_id):
        """Add a new task to all LoRA modules"""
        self.num_tasks = max(self.num_tasks, task_id + 1)
        for module in self.lora_modules:
            if hasattr(module, 'add_task'):
                module.add_task(task_id)
    
    def set_task(self, task_id):
        """Set the current task for all LoRA modules"""
        if task_id >= self.num_tasks:
            self.add_task(task_id)
        
        self.current_task = task_id
        for module in self.lora_modules:
            if hasattr(module, 'set_task'):
                module.set_task(task_id)
    
    def enable_lora_training(self, task_id=None):
        """
        Enable LoRA parameters for training while keeping original parameters frozen
        """
        if task_id is not None:
            self.set_task(task_id)
        
        for name, param in self.named_parameters():
            if (f'task{self.current_task}' in name and ('lora_A' in name or 'lora_B' in name)) or ('task' in name and 'scales' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def disable_lora_training(self):
        """
        Enable all parameters for training
        """
        for param in self.parameters():
            param.requires_grad = True
    
    def get_task_lora_parameters(self, task_id=None):
        """
        Get LoRA parameters for a specific task
        """
        if task_id is None:
            task_id = self.current_task
        
        task_params = []
        task_name = f'task{task_id}'
        
        for name, param in self.named_parameters():
            if task_name in name and ('lora_A' in name or 'lora_B' in name or 'scales' in name):
                if param.requires_grad:
                    task_params.append(param)
        
        return task_params
    
    def get_all_lora_parameters(self):
        """
        Get all LoRA parameters across all tasks
        """
        lora_params = []
        for name, param in self.named_parameters():
            if ('lora_A' in name or 'lora_B' in name or 'scales' in name) and param.requires_grad:
                lora_params.append(param)
        return lora_params
    
    def save_task_lora_weights(self, task_id, path):
        """
        Save LoRA weights for a specific task
        """
        task_state_dict = {}
        task_name = f'task{task_id}'
        
        for name, param in self.named_parameters():
            if task_name in name and ('lora_A' in name or 'lora_B' in name or 'scales' in name):
                task_state_dict[name] = param.data.clone()
        
        torch.save({
            'task_id': task_id,
            'lora_rank': self.lora_rank,
            'state_dict': task_state_dict
        }, path)
    
    def load_task_lora_weights(self, path):
        """
        Load LoRA weights for a specific task
        """
        checkpoint = torch.load(path)
        task_id = checkpoint['task_id']
        lora_rank = checkpoint['lora_rank']
        state_dict = checkpoint['state_dict']
        
        # Ensure task exists
        if task_id >= self.num_tasks:
            self.add_task(task_id)
        
        # Load weights
        current_state_dict = self.state_dict()
        current_state_dict.update(state_dict)
        self.load_state_dict(current_state_dict)
        
        return task_id
    
    def save_all_lora_weights(self, path):
        """
        Save all LoRA weights across all tasks
        """
        all_lora_state_dict = {}
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name or 'scales' in name:
                all_lora_state_dict[name] = param.data.clone()
        
        torch.save({
            'num_tasks': self.num_tasks,
            'lora_rank': self.lora_rank,
            'current_task': self.current_task,
            'state_dict': all_lora_state_dict
        }, path)
    
    def load_all_lora_weights(self, path):
        """
        Load all LoRA weights across all tasks
        """
        checkpoint = torch.load(path)
        num_tasks = checkpoint['num_tasks']
        lora_rank = checkpoint['lora_rank']
        current_task = checkpoint['current_task']
        state_dict = checkpoint['state_dict']
        
        # Ensure we have enough tasks
        for task_id in range(num_tasks):
            if task_id >= self.num_tasks:
                self.add_task(task_id)
        
        # Load weights
        current_state_dict = self.state_dict()
        current_state_dict.update(state_dict)
        self.load_state_dict(current_state_dict)
        
        # Set current task
        self.set_task(current_task)
    
    def freeze_task_lora(self, task_id):
        """
        Freeze LoRA parameters for a specific task
        """
        task_name = f'task{task_id}'
        for name, param in self.named_parameters():
            if task_name in name and ('lora_A' in name or 'lora_B' in name or 'scales' in name):
                param.requires_grad = False
    
    def unfreeze_task_lora(self, task_id):
        """
        Unfreeze LoRA parameters for a specific task
        """
        task_name = f'task{task_id}'
        for name, param in self.named_parameters():
            if task_name in name and ('lora_A' in name or 'lora_B' in name or 'scales' in name):
                param.requires_grad = True


# Usage example:
"""
Dynamic Multi-task LoRA Implementation

# Initialize model with LoRA configuration
args.lora_rank = 8      # LoRA rank
args.num_tasks = 1      # Initial number of tasks

model = Module(args, vocab)

# Training task 0
model.set_task(0)
model.enable_lora_training(task_id=0)
optimizer = torch.optim.Adam(model.get_task_lora_parameters(0), lr=1e-4)

# Add and train task 1
model.add_task(1)
model.set_task(1)
model.enable_lora_training(task_id=1)
optimizer = torch.optim.Adam(model.get_task_lora_parameters(1), lr=1e-4)

# Save/load specific task weights
model.save_task_lora_weights(0, 'task0_lora.pt')
model.save_task_lora_weights(1, 'task1_lora.pt')
model.load_task_lora_weights('task0_lora.pt')

# Switch between tasks during inference
model.set_task(0)  # Use task 0 LoRA layers
output0 = model(batch)

model.set_task(1)  # Use task 1 LoRA layers  
output1 = model(batch)

# Features:
# - Automatic traversal and replacement of linear/attention layers
# - Dynamic task addition without retraining
# - Independent LoRA parameters per task (named task{i})
# - Scalable to arbitrary number of tasks
# - Task-specific training and inference
"""
