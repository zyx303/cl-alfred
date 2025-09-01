import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Any, Optional

class BaseCLMethod:
    """
    Base class for continual learning methods
    Provides minimal implementation for compatibility with train_vlm
    """
    
    def __init__(self, n_classes: int, model: nn.Module, **kwargs):
        """
        Initialize base continual learning method
        
        Args:
            n_classes: Number of classes/tasks
            model: The neural network model
            **kwargs: Additional arguments from args
        """
        self.n_classes = n_classes
        self.model = model
        self.device = model.device
        
        # Extract common parameters from kwargs
        self.lr = kwargs.get('lr', 1e-4)
        # self.memory_size = kwargs.get('memory_size', 500)
        self.stream_seed = kwargs.get('stream_seed', 1)
        self.batch_size = kwargs.get('batch_size', 4)
        
        # Initialize exposed classes set
        self.exposed_classes = []
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=0.01
        )
        
        # Initialize scheduler if needed
        sched_name = kwargs.get('sched_name', 'default')
        if sched_name != 'default':
            self.scheduler = self._create_scheduler(sched_name)
        else:
            self.scheduler = None
        
        # Training statistics
        self.samples_seen = 0
        self.current_task = 0
        self.training_stats = defaultdict(list)
        
        # Memory placeholder (can be implemented by subclasses)
        # self.memory = None
        
    def _create_scheduler(self, sched_name: str):
        """Create learning rate scheduler"""
        if sched_name == 'step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        elif sched_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        else:
            return None
    
    def add_new_class(self, class_name: str):
        """
        Add a new class to the exposed classes set
        
        Args:
            class_name: Name of the new class
        """
        if class_name not in self.exposed_classes:
            self.exposed_classes.append(class_name)
        print(f"Added new class: {class_name}. Total classes: {len(self.exposed_classes)}")
    
    def online_before_task(self, task_id: int):
        """
        Hook called before starting a new task
        
        Args:
            task_id: ID of the current task
        """
        self.current_task = task_id
        print(f"Starting task {task_id}")
    
    def online_after_task(self, task_id: int):
        """
        Hook called after finishing a task
        
        Args:
            task_id: ID of the completed task
        """
        print(f"Completed task {task_id}")
        
        # Save task statistics
        if hasattr(self, 'training_stats'):
            self.training_stats[f'task_{task_id}'] = {
                'samples_seen': self.samples_seen,
                'exposed_classes': len(self.exposed_classes)
            }
    
    # def update_memory(self, sample: Dict[str, Any]):
    #     """
    #     Update memory with new sample (placeholder implementation)
        
    #     Args:
    #         sample: Training sample to potentially store in memory
    #     """
    #     # Base implementation does nothing
    #     # Subclasses can implement actual memory management
    #     pass
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss (placeholder implementation)
        
        Returns:
            Regularization loss tensor (zero for base class)
        """
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def update_fisher_and_score(self, new_params: Dict, old_params: Dict, 
                               new_grads: Dict, old_grads: Dict):
        """
        Update Fisher information and importance scores (placeholder)
        
        Args:
            new_params: New model parameters
            old_params: Old model parameters  
            new_grads: New gradients
            old_grads: Old gradients
        """
        # Base implementation does nothing
        # Subclasses like EWC can implement Fisher information updates
        pass
    
    def update_schedule(self):
        """Update learning rate scheduler"""
        if self.scheduler is not None:
            self.scheduler.step()
    
    def report_training(self, sample_num: int, train_loss: Dict[str, float]):
        """
        Report training progress
        
        Args:
            sample_num: Number of samples seen so far
            train_loss: Dictionary of training losses
        """
        self.samples_seen = sample_num
        
        # Store training statistics
        self.training_stats['losses'].append(train_loss)
        
        # Print progress periodically
        if sample_num % 100 == 0:
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_loss.items()])
            print(f"Samples: {sample_num}, {loss_str}")
    
    def get_memory_batch(self, batch_size: int) -> Dict[str, Any]:
        """
        Get batch from memory (placeholder implementation)
        
        Args:
            batch_size: Size of memory batch to retrieve
            
        Returns:
            Dictionary containing memory batch data
        """
        # Base implementation returns empty batch
        return {'batch': []}
    
    def save_checkpoint(self, filepath: str):
        """
        Save method-specific checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'exposed_classes': list(self.exposed_classes),
            'samples_seen': self.samples_seen,
            'current_task': self.current_task,
            'training_stats': dict(self.training_stats),
            'optimizer_state': self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """
        Load method-specific checkpoint
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.exposed_classes = set(checkpoint.get('exposed_classes', []))
        self.samples_seen = checkpoint.get('samples_seen', 0)
        self.current_task = checkpoint.get('current_task', 0)
        self.training_stats = defaultdict(list, checkpoint.get('training_stats', {}))
        
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
        if 'scheduler_state' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
