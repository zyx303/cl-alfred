import os
import torch
import numpy as np
import nn.vnn as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
from PIL import Image
import concurrent.futures
from typing import List
import gen.constants as constants

try:
    from models.nn.llama_encoder import LlamaTextEncoder, LlamaEncoderConfig
    _HAS_LLaMA = True
except Exception:
    _HAS_LLaMA = False

try:
    from models.nn.clip import _CLIPViTL14_336
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False
# from utils.download_feat import download_hf_patterns

import constants
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']



class Module(Base):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # choose language encoder: LSTM (default) or LLaMA-2-7B
        self.use_llama = getattr(args, 'use_llama', False)
        self.use_clip = getattr(args, 'use_clip', False)
        if self.use_clip:
            assert _HAS_CLIP, "CLIP not available; please ensure openai-clip is installed."
            self.visual_model = _CLIPViTL14_336(args, eval=True)
        
        if self.use_llama:
            assert _HAS_LLaMA, "Transformers LLaMA encoder not available; please ensure transformers is installed."
            # 使用本地模型路径
            llama_model_name = getattr(args, 'llama_model_name', './initial_model/llama')
            llama_dtype = getattr(args, 'llama_dtype', 'float16')
            llama_max_length = int(getattr(args, 'llama_max_length', 512))
            device = 'cuda' if args.gpu else 'cpu'
            dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}.get(str(llama_dtype).lower(), torch.float16)
            self.llama = LlamaTextEncoder(LlamaEncoderConfig(model_name_or_path=llama_model_name, device=device, dtype=dtype, max_length=llama_max_length))
            llama_hidden = int(getattr(getattr(self.llama.model, 'config', None), 'hidden_size', 4096))
            self.proj_goal = nn.Linear(llama_hidden, args.dhid*2)
            self.proj_instr = nn.Linear(llama_hidden, args.dhid*2)
            self.enc_att_goal = vnn.SelfAttn(args.dhid*2)
            self.enc_att_instr = vnn.SelfAttn(args.dhid*2)
        else:
            # encoder and self-attention (legacy LSTM)
            self.enc_goal = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
            self.enc_instr = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
            self.enc_att_goal = vnn.SelfAttn(args.dhid*2)
            self.enc_att_instr = vnn.SelfAttn(args.dhid*2)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        self.dec = vnn.ABP(self.emb_action_low, args.dframe, 2*args.dhid,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # bce reconstruction loss
        # self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv_panoramic.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()

        self.panoramic = args.panoramic
        self.orientation = args.orientation

        # 并行 I/O 开关与并行度（无缓存）
        self.enable_io_parallel = getattr(args, 'enable_io_parallel', True)
        self.num_io_workers = int(getattr(args, 'num_io_workers', max(1, (os.cpu_count()//4 or 4))))


    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        # 先做非 I/O 的部分，并记录要加载的文件路径与有效帧数
        to_load = []  # list[(path, valid_len)]
        for ex, swapColor in batch:
            ###########
            # auxillary
            ###########

            if not self.test_mode:
                # progress monitor supervision
                if self.args.pm_aux_loss_wt > 0:
                    num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                    subgoal_progress = [(i+1)/float(num_actions) for i in range(num_actions)]
                    feat['subgoal_progress'].append(subgoal_progress)

            #########
            # inputs
            #########

            # serialize segments
            self.serialize_lang_action(ex)

            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_instr'], ex['num']['lang_instr']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr

            # append goal + instr
            feat['lang_goal'].append(lang_goal)
            feat['lang_instr'].append(lang_instr)
            if self.use_llama:
                try:
                    r_idx = ex['ann']['repeat_idx']
                    goal_text = ex['turk_annotations']['anns'][r_idx]['task_desc']
                    instr_text = ' '.join(ex['turk_annotations']['anns'][r_idx]['high_descs'])
                except Exception:
                    # fallback: join tokens if original text missing
                    goal_text = ' '.join(ex.get('ann', {}).get('goal', []))
                    instr_lists = ex.get('ann', {}).get('instr', [])
                    instr_text = ' '.join([' '.join(x) for x in instr_lists])
                feat.setdefault('lang_goal_text', []).append(goal_text)
                feat.setdefault('lang_instr_text', []).append(instr_text)

            #########
            # outputs
            #########

            if not self.test_mode:
                # low-level action
                feat['action_low'].append([a['action'] for a in ex['num']['action_low']])

                # low-level action mask
                if load_mask:
                    indices = []
                    for a in ex['plan']['low_actions']:
                        if a['api_action']['action'] in ['MoveAhead', 'LookUp', 'LookDown', 'RotateRight', 'RotateLeft']:
                            continue
                        if a['api_action']['action'] == 'PutObject':
                            label = a['api_action']['receptacleObjectId'].split('|')
                        else:
                            label = a['api_action']['objectId'].split('|')
                        indices.append(classes.index(label[4].split('_')[0] if len(label) >= 5 else label[0]))
                    feat['action_low_mask_label'].append(indices)
                    feat['action_low_mask_label_unflattened'].append(indices)

                # low-level valid interact
                feat['action_low_valid_interact'].append([a['valid_interact'] for a in ex['num']['action_low']])

            if self.use_clip:
                if load_frames and not self.test_mode:
                    root = self.get_task_root(ex) # eg : data/json_feat_2.1.0/train/look_at_obj_in_light-KeyChain-None-FloorLamp-223/trial_T20190909_110146_114031
                    image_folder = '/data/yongxi/image/'
                    #/home/yongxi/work/cl-alfred/data/json_feat_2.1.0/train/look_at_obj_in_light/look_at_obj_in_light-AlarmClock-None-DeskLamp-301/trial_T20190907_174127_043461/high_res_images_panoramic
                    image_root = root.replace('data/json_feat_2.1.0/', image_folder) 
                    image_root = image_root.replace('train/', f"train/{ex['task_type']}/") 
                    image_root = os.path.join(image_root, 'high_res_images_panoramic')
                    
                    # Panoramic directions: left, up, front, down, right
                    ds = ['left', 'up', 'front', 'down', 'right']

                    # Load and process images for each direction
                    imgs = {}
                    if os.path.exists(image_root):
                        image_files = sorted(os.listdir(image_root))
                        for i, d in enumerate(ds):
                            direction_imgs = []
                            for p in image_files:
                                img_path = os.path.join(image_root, p)
                                if os.path.exists(img_path):
                                    # Load image and crop to get direction-specific view
                                    img = Image.open(img_path)
                                    # Crop panoramic image: each direction is 300px wide
                                    cropped_img = img.crop((i*300, 0, (i+1)*300, 300))
                                    direction_imgs.append(cropped_img)
                            imgs[d] = direction_imgs
                    else:
                        # Fallback: create dummy images if path doesn't exist
                        dummy_img = Image.new('RGB', (300, 300), color='black')
                        for d in ds:
                            imgs[d] = [dummy_img] * max(1, len(feat['action_low'][-1]) if feat['action_low'] else 1)

                    # Encode images using CLIP
                    with torch.no_grad():
                        features = {}
                        for direction, img_list in imgs.items():
                            if img_list:
                                # Encode images in batches
                                feat_list = self.visual_model.encode_image(img_list)
                                features[direction] = feat_list
                            else:
                                # Create dummy features if no images
                                device = torch.device('cuda' if self.args.gpu else 'cpu')
                                features[direction] = torch.zeros((1, self.visual_model.output_channels), device=device)

                    # Assign features to appropriate frame types
                    action_len = len(feat['action_low'][-1]) if feat['action_low'] else 1
                    feat['frames'].append(features['front'][:action_len])
                    feat['frames_left'].append(features['left'][:action_len])
                    feat['frames_up'].append(features['up'][:action_len])
                    feat['frames_down'].append(features['down'][:action_len])
                    feat['frames_right'].append(features['right'][:action_len])
            else:
                if load_frames and not self.test_mode:
                    root = self.get_task_root(ex)
                    # if swapColor in [0]:
                    im = torch.load(os.path.join(root, self.feat_pt))
                    # elif swapColor in [1, 2]:
                    #     im = torch.load(os.path.join(root, 'feat_conv_colorSwap{}_panoramic.pt'.format(swapColor)))
                    # elif swapColor in [3, 4, 5, 6]:
                    #     im = torch.load(os.path.join(root, 'feat_conv_onlyAutoAug{}_panoramic.pt'.format(swapColor - 2)))
                    
                    feat['frames'].append(im[2][:len(feat['action_low'][-1])])

                    feat['frames_left'].append(im[0][:len(feat['action_low'][-1])])
                    feat['frames_up'].append(im[1][:len(feat['action_low'][-1])])
                    feat['frames_down'].append(im[3][:len(feat['action_low'][-1])])
                    feat['frames_right'].append(im[4][:len(feat['action_low'][-1])]) 



        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal', 'lang_instr'}:
                # language embedding and padding
                if self.use_llama:
                    # in llama mode we store texts separately; skip packing here
                    continue
                else:
                    seqs = [torch.tensor(vv, device=device) for vv in v]
                    pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                    seq_lengths = np.array(list(map(len, v)))
                    embed_seq = self.emb_word(pad_seq)
                    packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                    feat[k] = packed_input
            elif k in {'action_low_mask'}:
                # mask padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'action_low_mask_label'}:
                # label
                seqs = torch.tensor([vvv for vv in v for vvv in vv], device=device, dtype=torch.long)
                feat[k] = seqs
            elif k in {'action_low_mask_label_unflattened'}:
                # label
                seqs = [torch.tensor(vv, device=device, dtype=torch.long) for vv in v]
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            elif k in {'lang_goal_text', 'lang_instr_text'}:
                # keep raw texts as Python list for LLaMA tokenizer
                continue
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if ('frames' in k or 'orientation' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        return feat


    def serialize_lang_action(self, feat):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
            if not self.test_mode:
                feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]


    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask


    def forward(self, feat, max_decode=300):
        cont_lang_goal, enc_lang_goal = self.encode_lang(feat)
        cont_lang_instr, enc_lang_instr = self.encode_lang_instr(feat)
        state_0_goal = cont_lang_goal, torch.zeros_like(cont_lang_goal)
        state_0_instr = cont_lang_instr, torch.zeros_like(cont_lang_instr)

        frames = self.vis_dropout(feat['frames'])
        if self.panoramic:
            frames_left = self.vis_dropout(feat['frames_left'])
            frames_up = self.vis_dropout(feat['frames_up'])
            frames_down = self.vis_dropout(feat['frames_down'])
            frames_right = self.vis_dropout(feat['frames_right'])
            res = self.dec(enc_lang_goal, enc_lang_instr, frames, frames_left, frames_up, frames_down, frames_right, max_decode=max_decode, gold=feat['action_low'], state_0_goal=state_0_goal, state_0_instr=state_0_instr)
        else:
            res = self.dec(enc_lang_goal, enc_lang_instr, frames, max_decode=max_decode, gold=feat['action_low'], state_0_goal=state_0_goal, state_0_instr=state_0_instr)
        feat.update(res)
        return feat


    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        if self.use_llama:
            texts: List[str] = feat.get('lang_goal_text', [])
            if not texts:
                # fallback: rebuild from tokens if not provided
                texts = [''] * len(feat.get('frames', []))
            enc_lang = self.llama.encode(texts)  # [B, T, H_llama]
            enc_lang = self.lang_dropout(self.proj_goal(enc_lang))  # [B, T, 2*dhid]
            cont_lang = self.enc_att_goal(enc_lang)
            return cont_lang, enc_lang
        else:
            emb_lang = feat['lang_goal']
            self.lang_dropout(emb_lang.data)
            enc_lang, _ = self.enc_goal(emb_lang)
            enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
            self.lang_dropout(enc_lang)
            cont_lang = self.enc_att_goal(enc_lang)
            return cont_lang, enc_lang

    def encode_lang_instr(self, feat):
        '''
        encode goal+instr language
        '''
        if self.use_llama:
            texts: List[str] = feat.get('lang_instr_text', [])
            if not texts:
                texts = [''] * len(feat.get('frames', []))
            enc_lang = self.llama.encode(texts)
            enc_lang = self.lang_dropout(self.proj_instr(enc_lang))
            cont_lang = self.enc_att_instr(enc_lang)
            return cont_lang, enc_lang
        else:
            emb_lang = feat['lang_instr']
            self.lang_dropout(emb_lang.data)
            enc_lang, _ = self.enc_instr(emb_lang)
            enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
            self.lang_dropout(enc_lang)
            cont_lang = self.enc_att_instr(enc_lang)
            return cont_lang, enc_lang


    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t_goal': None,
            'state_t_instr': None,
            'e_t': None,
            'cont_lang_goal': None,
            'enc_lang_goal': None,
            'cont_lang_instr': None,
            'enc_lang_instr': None,
        }


    def step(self, feat, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features (goal)
        if self.r_state['cont_lang_goal'] is None and self.r_state['enc_lang_goal'] is None:
            self.r_state['cont_lang_goal'], self.r_state['enc_lang_goal'] = self.encode_lang(feat)

        # encode language features (instr)
        if self.r_state['cont_lang_instr'] is None and self.r_state['enc_lang_instr'] is None:
            self.r_state['cont_lang_instr'], self.r_state['enc_lang_instr'] = self.encode_lang_instr(feat)

        # initialize embedding and hidden states (goal)
        if self.r_state['state_t_goal'] is None:
            self.r_state['state_t_goal'] = self.r_state['cont_lang_goal'], torch.zeros_like(self.r_state['cont_lang_goal'])

        # initialize embedding and hidden states (instr)
        if self.r_state['e_t'] is None and self.r_state['state_t_instr'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang_instr'].size(0), 1)
            self.r_state['state_t_instr'] = self.r_state['cont_lang_instr'], torch.zeros_like(self.r_state['cont_lang_instr'])

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        # decode and save embedding and hidden states
        if self.panoramic:
            out_action_low, out_action_low_mask, state_t_goal, state_t_instr, \
            lang_attn_t_goal, lang_attn_t_instr, *_ = \
                self.dec.step(
                    self.r_state['enc_lang_goal'],
                    self.r_state['enc_lang_instr'],
                    feat['frames'][:, 0],
                    feat['frames_left'][:, 0],
                    feat['frames_up'][:, 0],
                    feat['frames_down'][:, 0],
                    feat['frames_right'][:, 0],
                    e_t=e_t,
                    state_tm1_goal=self.r_state['state_t_goal'],
                    state_tm1_instr=self.r_state['state_t_instr'],
                )
        else:
            out_action_low, out_action_low_mask, state_t_goal, state_t_instr, \
            lang_attn_t_goal, lang_attn_t_instr, *_ = \
                self.dec.step(
                    self.r_state['enc_lang_goal'],
                    self.r_state['enc_lang_instr'],
                    feat['frames'][:, 0],
                    e_t=e_t,
                    state_tm1_goal=self.r_state['state_t_goal'],
                    state_tm1_instr=self.r_state['state_t_instr'],
                )

        # save states
        self.r_state['state_t_goal'] = state_t_goal
        self.r_state['state_t_instr'] = state_t_instr
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)

        return feat


    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for (ex, _), alow, alow_mask in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask']):
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]

            # index to API actions
            words = self.vocab['action_low'].index2word(alow)

            p_mask = [alow_mask[t].detach().cpu().numpy() for t in range(alow_mask.shape[0])]

            pred[self.get_task_and_ann_id(ex)] = {
                'action_low': ' '.join(words),
                'action_low_mask': p_mask,
            }

        return pred


    def embed_action(self, action):
        '''
        embed low-level action
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        action_num = torch.tensor(self.vocab['action_low'].word2index(action), device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low'].view(-1, len(self.vocab['action_low']))
        l_alow = feat['action_low'].view(-1)
        p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']

        # action loss
        pad_valid = (l_alow != self.pad)
        alow_loss = self.ce_loss(p_alow, l_alow)
        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # mask loss
        valid_idxs = valid.view(-1).nonzero().view(-1)
        flat_p_alow_mask = p_alow_mask.view(p_alow_mask.shape[0] * p_alow_mask.shape[1], p_alow_mask.shape[2])[valid_idxs]
        losses['action_low_mask'] = self.ce_loss(flat_p_alow_mask, feat['action_low_mask_label']).mean() * self.args.mask_loss_wt

        # progress monitoring loss
        if self.args.pm_aux_loss_wt > 0:
            p_progress_goal = feat['out_progress_goal'].squeeze(2)
            p_progress_instr = feat['out_progress_instr'].squeeze(2)
            l_progress = feat['subgoal_progress']

            pg_loss_goal = self.mse_loss(p_progress_goal, l_progress)
            pg_loss_goal = pg_loss_goal.view(-1) * pad_valid.float() * valid.view(-1).float()

            pg_loss_instr = self.mse_loss(p_progress_instr, l_progress)
            pg_loss_instr = pg_loss_instr.view(-1) * pad_valid.float()

            progress_loss = pg_loss_goal.mean() + pg_loss_instr.mean()

            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss

        return losses

    def compute_loss_unflattened(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low']
        l_alow = feat['action_low']
        p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']

        # action loss
        pad_valid = (l_alow != self.pad)
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # mask loss
        valid_idxs = valid.view(-1).nonzero().view(-1)
        flat_p_alow_mask = p_alow_mask.view(p_alow_mask.shape[0] * p_alow_mask.shape[1], p_alow_mask.shape[2])[valid_idxs]
        losses['action_low_mask'] = self.ce_loss(flat_p_alow_mask, feat['action_low_mask_label']) * self.args.mask_loss_wt

        # progress monitoring loss
        if self.args.pm_aux_loss_wt > 0:
            p_progress_goal = feat['out_progress_goal'].squeeze(2)
            p_progress_instr = feat['out_progress_instr'].squeeze(2)
            l_progress = feat['subgoal_progress']

            pg_loss_goal = self.mse_loss(p_progress_goal, l_progress)
            pg_loss_goal = pg_loss_goal.view(-1) * pad_valid.float() * valid.view(-1).float()

            pg_loss_instr = self.mse_loss(p_progress_instr, l_progress)
            pg_loss_instr = pg_loss_instr.view(-1) * pad_valid.float()

            progress_loss = pg_loss_goal.mean() + pg_loss_instr.mean()

            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss

        return losses


    def compute_loss_unsummed(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = []
        for i in range(len(batch)):
            _losses = []

            # GT and predictions
            p_alow = out['out_action_low'][i].view(-1, len(self.vocab['action_low']))
            l_alow = feat['action_low'][i].view(-1)
            p_alow_mask = out['out_action_low_mask'][i]
            valid = feat['action_low_valid_interact'][i]

            # action loss
            pad_valid = (l_alow != self.pad)
            alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
            alow_loss *= pad_valid.float()
            alow_loss = alow_loss.mean()
            _losses.append(alow_loss * self.args.action_loss_wt)

            # mask loss
            valid_idxs = valid.view(-1).nonzero().view(-1)
            flat_p_alow_mask = p_alow_mask[valid_idxs]
            _losses.append(F.cross_entropy(flat_p_alow_mask, feat['action_low_mask_label_unflattened'][i]) * self.args.mask_loss_wt)

            # progress monitoring loss
            if self.args.pm_aux_loss_wt > 0:
                p_progress_goal = feat['out_progress_goal'][i].squeeze(1)
                p_progress_instr = feat['out_progress_instr'][i].squeeze(1)
                l_progress = feat['subgoal_progress'][i]

                pg_loss_goal = F.mse_loss(p_progress_goal, l_progress, reduction='none')
                pg_loss_goal = pg_loss_goal.view(-1) * pad_valid.float() * valid.view(-1).float()

                pg_loss_instr = F.mse_loss(p_progress_instr, l_progress, reduction='none')
                pg_loss_instr = pg_loss_instr.view(-1) * pad_valid.float()

                progress_loss = pg_loss_goal.mean() + pg_loss_instr.mean()
                _losses.append(progress_loss)

            losses.append(sum(_losses))
        losses = torch.stack(losses, dim=0) # np.array(losses)
        return losses


    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for (task, _) in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))

        return {k: sum(v)/len(v) for k, v in m.items()}
