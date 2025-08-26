import torch
from torch import nn
from torch.nn import functional as F

from utils.data_loader import StreamDataset, MemoryDataset
from methods.er_baseline import ER


class OLoRA(ER):
	"""
	O-LoRA method for continual learning in CL-ALFRED.

	This leverages the existing LoRA-enabled model in `models/model/seq2seq_im_mask_lora.py`.
	We treat each new class as a new task and:
	  - call model.add_task(new_task_id) to allocate fresh LoRA A/B for that task
	  - optimize only the current task's LoRA params (base frozen by the LoRA wrapper)
	  - on task completion, call model.complete_task(task_id) to decompose to directions
	"""

	def __init__(self, n_classes, model, **kwargs):
		super().__init__(n_classes, model, **kwargs)

		self.lora_rank = kwargs.get("lora_rank", 8)
		self.ortho_reg_weight = kwargs.get("ortho_reg_weight", 0.0)
		self.adaptation_lr = kwargs.get("adaptation_lr", self.lr)

		# Track mapping between class name and task id
		self.task_id_mapping = {}
		self.current_task_id = -1
		self.completed_tasks = set()

		# Rebuild optimizer to only update LoRA parameters if possible
		self._build_lora_optimizer()

	def _iter_lora_params(self):
		"""Yield LoRA parameters of the current task if model exposes LoRALinear modules."""
		current_task = f"task_{self.current_task_id}"
		for m in self.model.modules():
			if hasattr(m, 'lora_A_dict') and hasattr(m, 'lora_B_dict'):
				if current_task in m.lora_A_dict:
					yield from m.lora_A_dict[current_task].parameters()
				if current_task in m.lora_B_dict:
					yield from m.lora_B_dict[current_task].parameters()
				# also scales for current task
				if hasattr(m, 'scales') and current_task in m.scales:
					p = m.scales[current_task]
					if isinstance(p, nn.Parameter):
						yield p

	def _build_lora_optimizer(self):
		params = list(self._iter_lora_params())
		if len(params) == 0:
			# fallback to base optimizer
			self.lora_optimizer = None
			return
		self.lora_optimizer = torch.optim.Adam(params, lr=self.adaptation_lr)

	def _set_current_task_by_class(self, class_name):
		# Map class to task id
		if class_name not in self.task_id_mapping:
			new_tid = len(self.task_id_mapping)
			self.task_id_mapping[class_name] = new_tid
			# allocate new LoRA task in model if available
			if hasattr(self.model, 'add_task'):
				self.model.add_task(new_tid)
		self.current_task_id = self.task_id_mapping[class_name]
		# rebuild optimizer for the new task
		self._build_lora_optimizer()

	def add_new_class(self, class_name):
		# complete previous task before switching
		if self.current_task_id >= 0 and self.current_task_id not in self.completed_tasks:
			if hasattr(self.model, 'complete_task'):
				self.model.complete_task(self.current_task_id)
			self.completed_tasks.add(self.current_task_id)

		super().add_new_class(class_name)
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

			# zero grad both optimizers if present
			self.optimizer.zero_grad()
			if self.lora_optimizer is not None:
				self.lora_optimizer.zero_grad()

			loss = self.model.compute_loss(out, batch, feat)
			sum_loss = sum(loss.values())

			# optional orthogonal reg on current task LoRA
			if self.ortho_reg_weight > 0:
				sum_loss = sum_loss + self.ortho_reg_weight * self._orthogonal_regularization()

			if 'cls_loss' not in info:
				info['cls_loss'] = 0
			info['cls_loss'] += sum_loss.item()

			sum_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

			# prefer lora optimizer if exists, otherwise fall back
			if self.lora_optimizer is not None:
				self.lora_optimizer.step()
			else:
				self.optimizer.step()
			self.update_schedule()

		info = {k: v / iterations for k, v in info.items()}
		return info

	def _orthogonal_regularization(self):
		"""Encourage current LoRA delta weights to be orthogonal to past directions."""
		loss = 0.0
		cnt = 0
		current_task = f"task_{self.current_task_id}"
		for m in self.model.modules():
			if hasattr(m, 'lora_A_dict') and hasattr(m, 'lora_B_dict') and current_task in m.lora_A_dict:
				A = m.lora_A_dict[current_task].weight  # [r, in]
				B = m.lora_B_dict[current_task].weight  # [out, r]
				delta = B @ A  # [out, in]
				# compare with stored directions of previous tasks if any
				if hasattr(m, 'directions') and isinstance(m.directions, nn.ParameterDict):
					for name, D in m.directions.items():  # D: [out, in]
						if D is None or D.numel() == 0:
							continue
						# cosine similarity penalty
						sim = F.cosine_similarity(delta.flatten(), D.flatten(), dim=0)
						loss = loss + sim.abs()
						cnt += 1
		return loss / max(cnt, 1)

	def online_after_task(self, cur_iter):
		super().online_after_task(cur_iter)
		if self.current_task_id >= 0 and self.current_task_id not in self.completed_tasks:
			if hasattr(self.model, 'complete_task'):
				self.model.complete_task(self.current_task_id)
			self.completed_tasks.add(self.current_task_id)
		# rebuild optimizer so that on next task switch we pick the right params
		self._build_lora_optimizer()
