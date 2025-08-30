import torch
from torch import nn
from torch.nn import functional as F

from utils.data_loader import StreamDataset, MemoryDataset
from methods.er_baseline import ER
from torch.utils.tensorboard import SummaryWriter

class OLoRA():
	"""
	O-LoRA method for continual learning in CL-ALFRED.

	This leverages the existing LoRA-enabled model in `models/model/seq2seq_im_mask_lora.py`.
	We treat each new class as a new task and:
	  - call model.add_task(new_task_id) to allocate fresh LoRA A/B for that task
	  - optimize only the current task's LoRA params (base frozen by the LoRA wrapper)
	  - on task completion, call model.complete_task(task_id) to decompose to directions
	"""

	def __init__(self, n_classes, model, **kwargs):
		self.num_learned_class = 0
		self.num_learning_class = 1
		self.exposed_classes = []
		self.n_classes = n_classes
		self.model = model
		
		self.lora_rank = kwargs.get("lora_rank", 8)
		self.adaptation_lr = kwargs.get("adaptation_lr", 1e-3)
		# O-LoRA regularization weights
		self.lamda_1 = kwargs.get("lamda_1", 0.5)  # orthogonal
		self.lamda_2 = kwargs.get("lamda_2", 0.0)  # L2 on loranew

		# Track mapping between class name and task id
		self.task_id_mapping = {}
		self.current_task_id = -1
		self.completed_tasks = set()

		self.lr = kwargs["lr"]

		# 创建统一优化器：LoRA 参数使用 adaptation_lr，其他参数使用 lr
		self._build_unified_optimizer()
		self.lr_gamma = 0.9995
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_gamma)

		# Logger
		self.writer = SummaryWriter(log_dir=kwargs['dout'])

	def _iter_lora_params(self):
		"""Yield only O-LoRA trainable params (loranew_*)."""
		for name, p in self.model.named_parameters():
			if 'loranew_' in name:
				yield p

	def _build_unified_optimizer(self):
		"""创建统一优化器，对不同参数使用不同学习率。"""
		lora_params = []
		other_params = []
		
		for name, param in self.model.named_parameters():
			if param.requires_grad:
				if 'loranew_' in name:
					lora_params.append(param)
				else:
					other_params.append(param)
		
		# 构建参数组，不同组使用不同学习率
		param_groups = []
		if len(other_params) > 0:
			param_groups.append({'params': other_params, 'lr': self.lr})
		if len(lora_params) > 0:
			param_groups.append({'params': lora_params, 'lr': self.adaptation_lr})
		
		if len(param_groups) > 0:
			self.optimizer = torch.optim.Adam(param_groups)
		else:
			# 如果没有可训练参数，创建空的优化器
			self.optimizer = torch.optim.Adam([], lr=self.lr)

	def reset_opt(self):
		"""重建统一优化器与调度器。"""
		self._build_unified_optimizer()
		self.lr_gamma = 0.9995
		self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_gamma)

	def _set_current_task_by_class(self, class_name):
		# finalize previous and reset for new task
		# if hasattr(self.model, 'finalize_lora_task'):
		# 	self.model.finalize_lora_task()
		# Map class to pseudo task id for bookkeeping (not used by model)
		if class_name not in self.task_id_mapping:
			new_tid = len(self.task_id_mapping)
			self.task_id_mapping[class_name] = new_tid
		self.current_task_id = self.task_id_mapping[class_name]
		# reset loranew for a fresh start
		if hasattr(self.model, 'reset_loranew'):
			self.model.reset_loranew()
		# rebuild optimizer for trainable loranew params
		self._build_unified_optimizer()

	def add_new_class(self, class_name):
		# complete previous task before switching
		if self.current_task_id >= 0 and self.current_task_id not in self.completed_tasks:
			if hasattr(self.model, 'complete_task'):
				self.model.complete_task(self.current_task_id)
			self.completed_tasks.add(self.current_task_id)

		self.exposed_classes.append(class_name)
		self.num_learned_class = len(self.exposed_classes)
		self._set_current_task_by_class(class_name)

	def online_train(self, sample, batch_size, iterations=1, stream_batch_size=1):
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

			# 清零梯度
			self.optimizer.zero_grad()

			loss = self.model.compute_loss(out, batch, feat)
			sum_loss = sum(loss.values())

			# O-LoRA regularization
			orthogonal_loss, l2_loss = self._olora_regularization()
			sum_loss = sum_loss + self.lamda_1 * orthogonal_loss + self.lamda_2 * l2_loss

			if 'cls_loss' not in info:
				info['cls_loss'] = 0
			info['cls_loss'] += sum_loss.item()

			sum_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

			# 使用统一优化器更新参数
			self.optimizer.step()
			self.update_schedule()

		info = {k: v / iterations for k, v in info.items()}
		return info

	def _olora_regularization(self):
		"""O-LoRA regularization: sum |A @ Anew^T| and L2 on loranew params."""
		# collect A and Anew by module prefix
		lora_A, loranew_A = {}, {}
		for name, p in self.model.named_parameters():
			if name.endswith('lora_A.weight'):
				prefix = name.split('lora_A')[0]
				lora_A[prefix] = p
			elif name.endswith('loranew_A.weight'):
				prefix = name.split('loranew_A')[0]
				loranew_A[prefix] = p
		orthogonal_loss = 0.0
		for k, A in lora_A.items():
			if k in loranew_A:
				Anew = loranew_A[k]
				orthogonal_loss = orthogonal_loss + torch.abs(A @ Anew.t()).sum()
		# L2 on all loranew params
		l2_loss = 0.0
		for name, p in self.model.named_parameters():
			if 'loranew_' in name:
				l2_loss = l2_loss + torch.norm(p, p=2)
		return orthogonal_loss, l2_loss

	def regularization_loss(self):
		"""供离线训练路径调用的统一正则。"""
		orthogonal_loss, l2_loss = self._olora_regularization()
		return self.lamda_1 * orthogonal_loss + self.lamda_2 * l2_loss

	def report_training(self, sample_num, info):
		for k in info:
			self.writer.add_scalar(f'train/{k}', info[k], sample_num)

	def online_after_task(self, cur_iter):
		# super().online_after_task(cur_iter)
		if self.current_task_id >= 0 and self.current_task_id not in self.completed_tasks:
			if hasattr(self.model, 'finalize_lora_task'):
				self.model.finalize_lora_task()
			self.completed_tasks.add(self.current_task_id)
		# start next with fresh loranew
		if hasattr(self.model, 'reset_loranew'):
			self.model.reset_loranew()
		# rebuild optimizer so that on next task switch we pick the right params
		self._build_unified_optimizer()
