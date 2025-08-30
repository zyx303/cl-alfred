#!/bin/bash

# Eval O-LoRA checkpoint on CL-ALFRED
# bash eval_olora_s1.sh > exp/behavior_il_new/olora/s1/eval_1_on_1.log 2>&1
export ALFRED_ROOT=.
export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

CKPT_PATH=/home/yongxi/work/cl-alfred/exp/behavior_il_new/olora/s1/net_epoch_000022510_look_at_obj_in_light.pth
SPLIT=valid_seen

python models/eval/eval_seq2seq.py \
  --model_path ${CKPT_PATH} \
  --eval_split ${SPLIT} \
  --incremental_setup behavior_il \
  --incremental_type look_at_obj_in_light \
  --stream_seed 1 \
  --num_threads 6 \
  --x_display 0 \
  --gpu \
  --model models.model.seq2seq_im_mask_olora

