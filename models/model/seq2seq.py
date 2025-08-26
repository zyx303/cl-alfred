import os
import sys

import random
import json
import torch
import collections
import numpy as np
from torch import nn
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from utils.method_manager import select_method
from collections import defaultdict



class Module(nn.Module):

    def __init__(self, args, vocab):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__()

        # sentinel tokens
        self.pad = 0
        self.seg = 1

        # args and vocab
        self.args = args
        self.vocab = vocab

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)

        # end tokens
        self.stop_token = self.vocab['action_low'].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)

        # set random seed (Note: this is not the seed used to initialize THOR object locations)
        random.seed(a=args.seed)
        np.random.seed(args.seed)

    def run_train1(self, args=None):
        '''
        按任务的离线训练：每个 task 使用 DataLoader 批训练（可多 epoch），不再逐样本 online。
        '''

        # args
        args = args or self.args

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=4)

        # display dout
        print("Saving to: %s" % self.args.dout)

        cl_method = select_method(args=args, n_classes=args.n_tasks, model=self)

        # 验证集（保持原格式）
        test_datalist_seen = json.load(open(f'embodied_split/{args.incremental_setup}/valid_seen.json', 'r'))
        test_datalist_seen = [(s, False) for s in test_datalist_seen]
        test_datalist_unseen = json.load(open(f'embodied_split/{args.incremental_setup}/valid_unseen.json', 'r'))
        test_datalist_unseen = [(s, False) for s in test_datalist_unseen]

        # 训练配置
        # epochs_per_task = getattr(args, 'epochs_per_task', 5)
        epochs_per_task = 5
        batch_size = args.batchsize

        samples_cnt = 0
        for cur_iter in range(args.n_tasks):
            # 加载当前任务数据
            cur_train_datalist = json.load(open(
                f'embodied_split/{args.incremental_setup}/embodied_data_disjoint_rand{args.stream_seed}_cls1_task{cur_iter}.json', 'r'
            ))

            # 预先补齐每个样本的 frame 统计，供日志/可视化使用
            for d in cur_train_datalist:
                if 'num_frames' not in d:
                    traj_data = self.load_task_json(d['task'])
                    d['num_frames'] = len([aa for a in traj_data['num']['action_low'] for aa in a])

            # 若出现新类别，注册到方法管理器（影响 MemoryDataset 的标签映射）
            for d in cur_train_datalist:
                if d['klass'] not in cl_method.exposed_classes:
                    cl_method.add_new_class(d['klass'])

            # 同步 memory 的类别映射（即使 memory 为空，也需包含当前所有已曝光类别）
            if hasattr(cl_method, 'memory'):
                mem = cl_method.memory
                # 更新类名列表与映射
                mem.cls_list = list(cl_method.exposed_classes)
                mem.cls_dict = {mem.cls_list[i]: i for i in range(len(mem.cls_list))}
                # 确保计数结构长度匹配
                needed = len(mem.cls_list) - len(mem.cls_count)
                for _ in range(max(0, needed)):
                    mem.cls_count.append(0)
                    mem.cls_idx.append([])
                    mem.cls_train_cnt = np.append(mem.cls_train_cnt, 0)

            # 构建 DataLoader（stream 批次）
            from utils.data_loader import StreamDataset
            stream_ds = StreamDataset(datalist=cur_train_datalist,
                                      cls_list=cl_method.exposed_classes,
                                      data_dir=getattr(args, 'data_dir', None))
            stream_loader = DataLoader(stream_ds, batch_size=batch_size, shuffle=True,
                                       drop_last=False, collate_fn=lambda x: x)

            # 为了能把 (task, swapColor) 还原为完整 sample，这里建映射
            task_to_sample = {json.dumps(s['task'], sort_keys=True): s for s in cur_train_datalist}

            # 任务前钩子
            if hasattr(cl_method, 'online_before_task'):
                cl_method.online_before_task(cur_iter)

            # 多 epoch 训练
            for epoch in range(epochs_per_task):
                self.train()
                pbar = tqdm(stream_loader, desc=f"Task {cur_iter} Epoch {epoch+1}/{epochs_per_task}")
                for stream_batch in pbar:
                    # stream_batch: List[(task_dict, 0)]
                    stream_batch = list(stream_batch)
                    bs_stream = len(stream_batch)

                    # 组装最终 batch：stream + memory 回放
                    data = []
                    if bs_stream > 0:
                        data += stream_batch

                    memory_batch_size = max(0, batch_size - bs_stream)
                    if hasattr(cl_method, 'memory') and memory_batch_size > 0 and len(cl_method.memory) > 0:
                        memory_data = cl_method.memory.get_batch(memory_batch_size)
                        data += memory_data['batch']

                    # 前向 & 反向
                    batch = [(self.load_task_json(task), swapColor) for task, swapColor in data]
                    feat = self.featurize(batch)

                    out = self.forward(feat)
                    cl_method.optimizer.zero_grad()
                    loss_dict = self.compute_loss(out, batch, feat)
                    sum_loss = sum(loss_dict.values())

                    # 可选：方法级正则（如 EWC）
                    if hasattr(cl_method, 'regularization_loss'):
                        reg_loss = cl_method.regularization_loss()
                        sum_loss = sum_loss + reg_loss

                    # 记录旧参数/梯度（用于 EWC 的 Fisher/score 更新）
                    old_params = {n: p.clone().detach() for n, p in self.named_parameters() if p.requires_grad} if hasattr(cl_method, 'update_fisher_and_score') else None
                    old_grads = {n: (p.grad.clone().detach() if p.grad is not None else None) for n, p in self.named_parameters() if p.requires_grad} if hasattr(cl_method, 'update_fisher_and_score') else None

                    sum_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                    cl_method.optimizer.step()

                    # 可选：更新 Fisher 与 score
                    if hasattr(cl_method, 'update_fisher_and_score') and old_params is not None:
                        new_params = {n: p.clone().detach() for n, p in self.named_parameters() if p.requires_grad}
                        new_grads = {n: (p.grad.clone().detach() if p.grad is not None else None) for n, p in self.named_parameters() if p.requires_grad}
                        # 过滤 None 梯度
                        new_grads = {k: v for k, v in new_grads.items() if v is not None}
                        old_grads = {k: v for k, v in old_grads.items() if v is not None}
                        try:
                            cl_method.update_fisher_and_score(new_params, old_params, new_grads, old_grads)
                        except TypeError:
                            # 与具体方法签名不匹配则跳过
                            pass
                    if hasattr(cl_method, 'update_schedule'):
                        cl_method.update_schedule()

                    samples_cnt += bs_stream

                    # 训练日志（与 ER.report_training 对齐的 key）
                    if hasattr(cl_method, 'report_training'):
                        cl_method.report_training(samples_cnt, {'cls_loss': float(sum_loss.detach().cpu())})

                    # 更新记忆库（仅用 stream 样本）
                    if hasattr(cl_method, 'update_memory'):
                        for task, _ in stream_batch:
                            key = json.dumps(task, sort_keys=True)
                            sample = task_to_sample.get(key)
                            if sample is not None:
                                cl_method.update_memory(sample)

                # 每个 epoch 结束做一次评估
                if hasattr(cl_method, 'evaluation') and hasattr(cl_method, 'report_test'):
                    eval_seen = cl_method.evaluation(test_datalist_seen, samples_cnt, batch_size)
                    cl_method.report_test(samples_cnt, eval_seen, tag='valid_seen')
                    eval_unseen = cl_method.evaluation(test_datalist_unseen, samples_cnt, batch_size)
                    cl_method.report_test(samples_cnt, eval_unseen, tag='valid_unseen')

            # 任务后钩子
            if hasattr(cl_method, 'online_after_task'):
                cl_method.online_after_task(cur_iter)

            # 保存检查点
            last_klass = cur_train_datalist[-1]['klass'] if len(cur_train_datalist) else f'task{cur_iter}'
            torch.save({
                'metric': {'samples_cnt': samples_cnt},
                'model': self.state_dict(),
                'optim': cl_method.optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, os.path.join(args.dout, 'net_epoch_%09d_%s.pth' % (samples_cnt, last_klass)))


    def run_pred(self, dev, batch_size=32, name='dev', iter=0):
        '''
        validation loop
        '''
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter
        for batch, feat in self.iterate(dev, batch_size):
            out = self.forward(feat)
            preds = self.extract_preds(out, batch, feat)
            p_dev.update(preds)
            loss = self.compute_loss(out, batch, feat)
            for k, v in loss.items():
                ln = 'loss_' + k
                m_dev[ln].append(v.item())
            sum_loss = sum(loss.values())
            total_loss.append(float(sum_loss.detach().cpu()))
            dev_iter += len(batch)

        m_dev = {k: sum(v) / len(v) for k, v in m_dev.items()}
        total_loss = sum(total_loss) / len(total_loss)
        return p_dev, dev_iter, total_loss, m_dev

    def featurize(self, batch):
        raise NotImplementedError()

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def get_task_and_ann_id(self, ex):
        '''
        single string for task_id and annotation repeat idx
        '''
        return "%s_%s" % (ex['task_id'], str(ex['ann']['repeat_idx']))

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            debug[i] = {
                'lang_goal': ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc'],
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                'p_action_low': preds[i]['action_low'].split(),
            }
        return debug

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    def iterate(self, data, batch_size):
        '''
        breaks dataset into batch_size chunks for training
        '''
        for i in range(0, len(data), batch_size):
            tasks = data[i:i+batch_size]
            batch = [(self.load_task_json(task), swapColor) for task, swapColor in tasks]
            feat = self.featurize(batch)
            yield batch, feat

    def zero_input(self, x, keep_end_token=True):
        '''
        pad input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        return list(np.full_like(x[:-1], self.pad)) + end_token

    def zero_input_list(self, x, keep_end_token=True):
        '''
        pad a list of input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        lz = [list(np.full_like(i, self.pad)) for i in x[:-1]] + end_token
        return lz

    @staticmethod
    def adjust_lr(optimizer, init_lr, epoch, decay_epoch=5):
        '''
        decay learning rate every decay_epoch
        '''
        lr = init_lr * (0.1 ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @classmethod
    def load(cls, fsave):
        '''
        load pth model from disk
        '''
        save = torch.load(fsave)
        model = cls(save['args'], save['vocab'])
        model.load_state_dict(save['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(save['optim'])
        return model, optimizer

    @classmethod
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True
