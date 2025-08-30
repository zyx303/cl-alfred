#!/bin/bash

# O-LoRA training script for CL-ALFRED (uses LoRA-enabled seq2seq model)
# bash train_olora_s1.sh > exp/behavior_il_new/olora/s1/training.log 2>&1
export CUDA_VISIBLE_DEVICES=7
export ALFRED_ROOT=.

python models/train/train_seq2seq.py \
  --incremental_setup behavior_il \
  --mode olora \
  --stream_seed 1 \
  --dout exp/behavior_il_new/olora/s1 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --adaptation_lr 1e-3 \
  --lamda_1 0.5 \
  --lamda_2 0.0 \
  --eval_period 5000 \
  --batchsize 64 \
  --model_arch seq2seq_im_mask_olora
