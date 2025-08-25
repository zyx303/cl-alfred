import os
import re
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
import math
import constants
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']

class LoRALinear(nn.Module):
    """
    Linear layer with multi-task LoRA adaptation
    """
    def __init__(self, original_layer, rank=4, current_task=0, path=None):
        super().__init__()
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.lora_A_dict = nn.ModuleDict()
        self.lora_B_dict = nn.ModuleDict()
        self.task_id = current_task
        self.directions = nn.ParameterDict()
        self.scales = nn.ParameterDict()
        self._init_task_lora(current_task)

    # 不再从外部路径加载，全部依赖于 state_dict 恢复

        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def _load_previous_tasks(self, path):
        """占位：保留接口但不做任何加载（统一使用 state_dict）"""
        return
    
    def _init_task_lora(self, task_id):
        """初始化特定任务的LoRA参数"""
        task_name = f'task_{task_id}'
        
        # 创建新的LoRA A和B层
        lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        lora_B = nn.Linear(self.rank, self.out_features, bias=False)
        
        # 初始化权重
        nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(lora_B.weight)

        # 确保与原始层处于同一设备与dtype
        dev = self.original_layer.weight.device
        dtype = self.original_layer.weight.dtype
        lora_A = lora_A.to(device=dev, dtype=dtype)
        lora_B = lora_B.to(device=dev, dtype=dtype)
        
        # 存储到字典中
        self.lora_A_dict[task_name] = lora_A
        self.lora_B_dict[task_name] = lora_B
        
        # 添加scale参数
        self.scales[task_name] = nn.Parameter(torch.ones(1, device=dev, dtype=dtype))
        self.scales[task_name].requires_grad = True

    def forward(self, x):
        # 保留线性层输入用于LoRA残差
        input_x = x
        out = self.original_layer(x)

        # 叠加已完成任务的方向增益：input_x @ direction^T -> [B, out_features]
        if self.task_id > 0:
            for i in range(self.task_id):
                tname = f'task_{i}'
                if tname in self.directions and tname in self.scales:
                    # directions[tname]: [out_features, in_features]
                    # input_x: [B, in_features]
                    out = out + self.scales[tname] * (input_x @ self.directions[tname].transpose(0, 1))

        # 应用当前任务的LoRA
        current_task_name = f'task_{self.task_id}'
        if current_task_name in self.lora_A_dict:
            lora_A = self.lora_A_dict[current_task_name]
            lora_B = self.lora_B_dict[current_task_name]
            lora_output = lora_B(lora_A(input_x))
            out = out + self.scales[current_task_name] * lora_output

        return out
    
    def add_task(self, new_task_id):
        """添加新任务并初始化其LoRA参数"""
        if new_task_id == self.task_id + 1:
            # 初始化新任务的LoRA参数
            self._init_task_lora(new_task_id)
            self.task_id = new_task_id

    def complete_task(self):
        """完成当前任务：计算 direction，写入本模块参数（供 state_dict 保存）"""
        current_task_name = f'task_{self.task_id}'
        dev = self.original_layer.weight.device
        # 计算权重分解（W_delta = B @ A）并按行归一化得到方向矩阵 [out, in]
        w_delta = self.lora_B_dict[current_task_name].weight @ self.lora_A_dict[current_task_name].weight
        dir = w_delta / (torch.norm(w_delta, dim=1, keepdim=True) + 1e-12)
        # 将direction添加到模型中（冻结方向，仅训练scale）
        self.directions[current_task_name] = nn.Parameter(dir.to(dev), requires_grad=False)
        self.scales[current_task_name].requires_grad = True




def replace_linear_with_lora(module, rank=4, current_task=0, target_modules=None):
    """
    Recursively replace Linear layers with LoRA versions
    """
    if target_modules is None:
        target_modules = ['Linear']
    
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Replace with LoRA version
            lora_linear = LoRALinear(child, rank=rank, current_task=current_task, path=None)
            setattr(module, name, lora_linear)
        else:
            # Recursively apply to child modules
            replace_linear_with_lora(child, rank, current_task, target_modules)


def replace_attention_with_lora(module, rank=4, current_task=0):
    """
    Replace attention q, v projections with LoRA versions
    """
    for name, child in module.named_children():
        if hasattr(child, 'fc_key') and hasattr(child, 'fc_query'):
            # This looks like a scaled dot attention module - replace q,v (key,query)
            if isinstance(child.fc_key, nn.Linear):
                child.fc_key = LoRALinear(child.fc_key, rank=rank, current_task=current_task)
            if isinstance(child.fc_query, nn.Linear):
                child.fc_query = LoRALinear(child.fc_query, rank=rank, current_task=current_task)
        
        elif hasattr(child, 'scorer') and isinstance(child.scorer, nn.Linear):
            # This looks like a self attention module
            child.scorer = LoRALinear(child.scorer, rank=rank, current_task=current_task)
        
        else:
            # Recursively apply to child modules
            replace_attention_with_lora(child, rank, current_task)


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
        self.current_task = getattr(args, 'current_task', 1)
        self.current_task = 0
        # Apply LoRA to all linear layers and attention modules
        self._apply_lora_to_model()
        # Keep track of LoRA modules for task switching
        self.lora_modules = []
        # self._collect_lora_modules()
    
    def _apply_lora_to_model(self):
        """Apply LoRA to all linear layers and attention q,v projections"""
        # Replace linear layers with LoRA versions
        replace_linear_with_lora(self, rank=self.lora_rank, current_task=self.current_task)
        # Replace attention q,v projections with LoRA versions
        # replace_attention_with_lora(self, rank=self.lora_rank, current_task=self.current_task)
    
    # def _collect_lora_modules(self):
    #     """Collect all LoRA modules for easy task switching"""
    #     self.lora_modules = []
    #     for module in self.modules():
    #         if isinstance(module, LoRALinear):
    #             self.lora_modules.append(module)
    
    def add_task(self, task_id):
        """Add a new task to all LoRA modules"""
        self.current_task = max(self.current_task, task_id + 1)
        # 为所有LoRA模块添加新任务
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.add_task(task_id)
    
        print(f"Added task {task_id} with fresh LoRA parameters for all modules")
    
    def complete_task(self, task_id):
        """Complete a task: compute directions and store inside modules (state_dict covers all)."""
        print(f"Completing task {task_id} and performing weight decomposition...")
        for _, module in self.named_modules():
            if isinstance(module, LoRALinear):
                module.complete_task()
        print(f"Task {task_id} decomposition completed")

    @classmethod
    def load(cls, fsave):
        """
        Custom loader that prepares LoRA task placeholders before loading state_dict
        so checkpoints containing task_1/task_2 and directions/scales can be restored.
        """
        save = torch.load(fsave, map_location='cpu')
        # instantiate model first
        model = cls(save['args'], save['vocab'])

        # extract model state dict (support both keys)
        state = save.get('model', None)
        if state is None:
            state = save.get('model_state_dict', {})

        # discover all task ids present in checkpoint
        task_ids = set()
        task_pattern = re.compile(r"\.task_(\d+)\.")
        for key in state.keys():
            if any(s in key for s in [
                'lora_A_dict.task_', 'lora_B_dict.task_', 'scales.task_', 'directions.task_']):
                m = task_pattern.search(key)
                if m:
                    task_ids.add(int(m.group(1)))
        if not task_ids:
            # no extra tasks found; proceed with vanilla load
            model.load_state_dict(state, strict=False)
        else:
            max_tid = max(task_ids)
            # prepare placeholders for each LoRALinear module
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    # ensure lora/scales exist for all tasks up to max
                    for tid in range(0, max_tid + 1):
                        tname = f'task_{tid}'
                        if tname not in module.lora_A_dict:
                            module._init_task_lora(tid)
                        # ensure directions placeholder exists so it can be loaded
                        if tname not in module.directions:
                            dev = module.original_layer.weight.device
                            dtype = module.original_layer.weight.dtype
                            placeholder = torch.zeros(module.out_features, module.in_features, device=dev, dtype=dtype)
                            module.directions[tname] = nn.Parameter(placeholder, requires_grad=False)
                    # set task id to the highest available
                    module.task_id = max(module.task_id, max_tid)
            # now load with strict=False to be safe with any minor mismatches
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected:
                print(f"[LoRA load] Ignored unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
            if missing:
                print(f"[LoRA load] Missing keys count: {len(missing)}")

        # optimizer (if present)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        if 'optim' in save:
            try:
                optimizer.load_state_dict(save['optim'])
            except Exception as e:
                print(f"[LoRA load] Optimizer state load failed: {e}")
        return model, optimizer
    
    