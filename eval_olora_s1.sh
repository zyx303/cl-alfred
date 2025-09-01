#!/bin/bash

# Eval O-LoRA checkpoint on CL-ALFRED
# bash eval_olora_s1.sh > exp/behavior_il_new/olora/s1/eval_1_on_1.log 2>&1
export ALFRED_ROOT=.
export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0
export debug=0


CKPT_PATH=/home/yongxi/work/cl-alfred/exp/behavior_il_new/olora/s1/net_epoch_000051940_pick_heat_then_place_in_recep.pth
SPLIT=valid_seen

python models/eval/eval_seq2seq.py \
  --model_path ${CKPT_PATH} \
  --eval_split ${SPLIT} \
  --incremental_setup behavior_il \
  --incremental_type pick_heat_then_place_in_recep \
  --stream_seed 1 \
  --num_threads 4 \
  --x_display 0 \
  --gpu \
  --model models.model.seq2seq_im_mask_olora

