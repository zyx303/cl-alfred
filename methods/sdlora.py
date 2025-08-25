import torch
import torch.nn as nn
import numpy as np
import copy
import math
import os
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.nn import functional as F

from utils.data_loader import StreamDataset, MemoryDataset
from methods.er_baseline import ER

class SDLoRA(ER):
    """
    SD-LoRA (Structured Decomposition LoRA) for continual learning in CL-ALFRED
    
    This method uses the multi-task LoRA model for continual learning of embodied AI tasks.
    It leverages the dynamic task addition and task-specific LoRA parameters.
    """
    
    def __init__(self, n_classes, model, **kwargs):
        super().__init__(n_classes, model, **kwargs)
        
        # SD-LoRA specific parameters
        self.lora_rank = kwargs.get("lora_rank", 10)
        self.adaptation_lr = kwargs.get("adaptation_lr", 1e-4)
        self.ortho_reg_weight = kwargs.get("ortho_reg_weight", 0.1)
        self.task_id_mapping = {}  # Map class names to task IDs
        self.current_task_id = 0
        self.completed_tasks = set()  # Track completed tasks
        self.lora_optimizer = None
        
        # Setup optimizer for LoRA parameters only
        self._setup_lora_optimizer()
        
        print(f"SD-LoRA initialized with LoRA rank={self.lora_rank}")
        print(f"Model is LoRA enabled: {self._has_lora_modules()}")
    
    def _has_lora_modules(self):
        """Check if model has LoRA modules"""
        for module in self.model.modules():
            if hasattr(module, 'lora_A_dict') and hasattr(module, 'lora_B_dict'):
                return True
        return False
    
    def _setup_lora_optimizer(self):
        """Setup optimizer specifically for current task's LoRA parameters"""
        # Get LoRA parameters for current task
        lora_params = self._get_current_task_lora_parameters()
        
        if lora_params:
            self.lora_optimizer = torch.optim.Adam(lora_params, lr=self.adaptation_lr)
            print(f"Setup LoRA optimizer for task {self.current_task_id} with {len(lora_params)} parameters")
        else:
            self.lora_optimizer = None
            print("Warning: No LoRA parameters found for optimizer")
    
    def _get_current_task_lora_parameters(self):
        """Get current task's LoRA parameters from all LoRALinear modules"""
        lora_params = []
        current_task_name = f'task_{self.current_task_id}'
        
        for module in self.model.modules():
            if hasattr(module, 'lora_A_dict') and hasattr(module, 'lora_B_dict'):
                # This is a LoRALinear module
                if current_task_name in module.lora_A_dict:
                    lora_params.extend(list(module.lora_A_dict[current_task_name].parameters()))
                if current_task_name in module.lora_B_dict:
                    lora_params.extend(list(module.lora_B_dict[current_task_name].parameters()))
                if current_task_name in module.scales:
                    lora_params.append(module.scales[current_task_name])
        
        return lora_params
    
    
    def orthogonal_regularization(self):
        """Compute orthogonal regularization loss for current task's LoRA parameters"""
        ortho_loss = 0.0
        count = 0
        
        # Get current task's LoRA parameters
        current_task_name = f"task_{self.current_task_id}"
        
        for module in self.model.modules():
            if hasattr(module, 'lora_A_dict') and current_task_name in module.lora_A_dict:
                A = module.lora_A_dict[current_task_name].weight
                if A.size(0) > 1:  # Only if rank > 1
                    gram_matrix = torch.mm(A, A.t())
                    identity = torch.eye(A.size(0), device=A.device)
                    ortho_loss += torch.norm(gram_matrix - identity, 'fro') ** 2
                    count += 1
        
        return ortho_loss / max(count, 1)
    
    def add_new_class(self, class_name):
        """Handle new task/class by adding new LoRA task and setting up training"""
        # Complete and decompose previous task if it exists
        if self.current_task_id >= 0 and self.current_task_id not in self.completed_tasks:
            self._complete_current_task()
            self.completed_tasks.add(self.current_task_id)
        
        # Call parent method first
        super().add_new_class(class_name)
        
        # Determine new task ID
        new_task_id = len(self.exposed_classes) - 1
        
        # Add mapping from class name to task ID
        self.task_id_mapping[class_name] = new_task_id
        
        # Add new task to LoRA model
        self.model.add_task(new_task_id)
        
        # Update current task ID
        self.current_task_id = new_task_id
        
        # Setup new optimizer for the new task's LoRA parameters
        self._setup_lora_optimizer()
        
        print(f"Added new task {new_task_id} for class '{class_name}'")
        print(f"Previous tasks {list(self.completed_tasks)} have been decomposed and frozen")
    
    def _complete_current_task(self):
        """Complete current task and decompose weights"""
        if hasattr(self.model, 'complete_task'):
            self.model.complete_task(self.current_task_id)
    
    def _get_task_id_for_class(self, class_name):
        """Get task ID for given class name"""
        if class_name in self.task_id_mapping:
            return self.task_id_mapping[class_name]
        return 0  # Default to task 0
    
    def _set_current_task_from_batch(self, batch):
        """Set current task based on the classes in the batch"""
        # For simplicity, use the first sample's class to determine task
        if batch:
            first_sample = batch[0]
            if isinstance(first_sample, dict) and 'klass' in first_sample:
                class_name = first_sample['klass']
                if class_name in self.task_id_mapping:
                    task_id = self.task_id_mapping[class_name]
                    if task_id != self.current_task_id:
                        self.current_task_id = task_id
                        self._setup_lora_optimizer()
    
    def online_step(self, sample, sample_num):
        """Online learning step with LoRA adaptation"""
        if sample['klass'] not in self.exposed_classes :
            self.add_new_class(sample['klass'])
            if self.current_task_id != 0:
                info = self.online_train(
                    self.temp_batch, self.batch_size,
                    iterations=int(self.num_updates),
                    stream_batch_size=self.temp_batchsize
                )
                self.report_training(sample_num, info)
                # for stored_sample in self.temp_batch:
                #     self.update_memory(stored_sample)
                self.temp_batch = []
                self.num_updates -= int(self.num_updates)

        self.temp_batch.append(sample)
        self.num_updates += self.online_iter

        if len(self.temp_batch) == self.temp_batchsize:
            # Set task based on current batch
            self._set_current_task_from_batch(self.temp_batch)
            
            info = self.online_train(
                self.temp_batch, self.batch_size,
                iterations=int(self.num_updates),
                stream_batch_size=self.temp_batchsize
            )
            self.report_training(sample_num, info)
            # for stored_sample in self.temp_batch:
            #     self.update_memory(stored_sample)
            self.temp_batch = []
            self.num_updates -= int(self.num_updates)

            
    def online_train(self, sample, batch_size, iterations=1, stream_batch_size=1):
        """Training loop with SD-LoRA adaptation using multi-task LoRA model"""
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(
                datalist=sample,
                cls_list=self.exposed_classes,
                data_dir=self.data_dir
            )
        # if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
        #     memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

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
            
            # Add orthogonal regularization for current task's LoRA parameters
            # ortho_loss = self.orthogonal_regularization()
            # total_loss = task_loss + self.ortho_reg_weight * ortho_loss

            total_loss = task_loss

            if 'cls_loss' not in info:
                info['cls_loss'] = 0
            info['cls_loss'] += task_loss.item()
            
            # if 'ortho_loss' not in info:
            #     info['ortho_loss'] = 0
            # info['ortho_loss'] += ortho_loss.item()
            
            total_loss.backward()
            
            # Gradient clipping for LoRA parameters
            if self.lora_optimizer:
                lora_params = self._get_current_task_lora_parameters()
                if lora_params:
                    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            
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
        
        # Log current task ID
        self.writer.add_scalar('train/current_task_id', self.current_task_id, sample_num)
        
        # Log LoRA parameter norms for current task
        current_task_name = f"task_{self.current_task_id}"
        total_lora_norm = 0.0
        
        module_count = 0
        for module in self.model.modules():
            if hasattr(module, 'lora_A_dict') and hasattr(module, 'lora_B_dict'):
                if current_task_name in module.lora_A_dict:
                    a_norm = torch.norm(module.lora_A_dict[current_task_name].weight).item()
                    b_norm = torch.norm(module.lora_B_dict[current_task_name].weight).item()
                    scale = module.scales[current_task_name].item() if current_task_name in module.scales else 1.0
                    
                    total_lora_norm += a_norm + b_norm
                    self.writer.add_scalar(f'lora_norms/module_{module_count}_A', a_norm, sample_num)
                    self.writer.add_scalar(f'lora_norms/module_{module_count}_B', b_norm, sample_num)
                    self.writer.add_scalar(f'lora_scales/module_{module_count}_scale', scale, sample_num)
                    module_count += 1
        
        self.writer.add_scalar('lora_norms/total', total_lora_norm, sample_num)
        
        # Log number of tasks
        num_tasks = len(self.exposed_classes)
        self.writer.add_scalar('train/num_tasks', num_tasks, sample_num)
    
    def save_model(self, path):
        """Save model including all LoRA weights"""
        # Save base model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lora_optimizer_state_dict': self.lora_optimizer.state_dict() if self.lora_optimizer else None,
            'current_task_id': self.current_task_id,
            'task_id_mapping': self.task_id_mapping,
            'exposed_classes': self.exposed_classes,
            'num_tasks': len(self.exposed_classes),
            'lora_rank': self.lora_rank,
        }, path)
        
        print(f"Saved model to {path}")
    
    def load_model(self, path):
        """Load model including all LoRA weights"""
        checkpoint = torch.load(path)
        
        # Load base model (support both keys) and allow extra LoRA keys
        model_state = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
        self.model.load_state_dict(model_state, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore task information
        self.current_task_id = checkpoint['current_task_id']
        self.task_id_mapping = checkpoint['task_id_mapping']
        self.exposed_classes = checkpoint['exposed_classes']
        
        # Setup optimizer for current task
        self._setup_lora_optimizer()
        
        if checkpoint['lora_optimizer_state_dict'] and self.lora_optimizer:
            self.lora_optimizer.load_state_dict(checkpoint['lora_optimizer_state_dict'])
        
        num_tasks = checkpoint.get('num_tasks', len(self.exposed_classes))
        print(f"Loaded model from {path} with {num_tasks} tasks")
    
    
    def load_task_checkpoint(self, path):
        """Load checkpoint for a specific task from unified file"""
        if hasattr(self.model, 'load_task_weights'):
            task_data = self.model.load_task_weights(path)
            print(f"Loaded task weights from unified file: {os.path.join(path, 'lora_weights.pt')}")
            return len(task_data)  # Return number of tasks loaded
        return 0
    
    def online_after_task(self, cur_iter):
        """Save LoRA parameters and decompose weights after each task"""
        super().online_after_task(cur_iter)
        
        # Decompose current task if not already done
        if self.current_task_id not in self.completed_tasks:
            self._complete_current_task()
            self.completed_tasks.add(self.current_task_id)
        
        print(f"Task {cur_iter} completed:")
        print(f"  - Task {self.current_task_id} weights decomposed and saved to unified file")
        print(f"  - Completed tasks: {sorted(list(self.completed_tasks))}")
    
    def evaluation(self, test_list, sample_num, batch_size=32):
        """Evaluation with current LoRA adaptation"""
        eval_dict = {}
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Run evaluation
        with torch.no_grad():
            p_valid, _, total_valid_loss, m_valid = self.model.run_pred(
                test_list, batch_size=batch_size, iter=sample_num
            )
            eval_dict['cls_loss'] = float(total_valid_loss)
            eval_dict.update(self.model.compute_metric(p_valid, test_list))
        
        return eval_dict
    
    def get_task_info(self):
        """Get information about current tasks"""
        return {
            'current_task_id': self.current_task_id,
            'num_tasks': len(self.exposed_classes),
            'task_id_mapping': self.task_id_mapping,
            'exposed_classes': self.exposed_classes
        }
