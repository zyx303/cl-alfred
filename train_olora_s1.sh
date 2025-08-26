#!/bin/bash

# O-LoRA training script for CL-ALFRED (uses LoRA-enabled seq2seq model)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export ALFRED_ROOT=.

python models/train/train_seq2seq.py \
  --incremental_setup behavior_il \
  --mode olora \
  --stream_seed 1 \
  --dout exp/behavior_il/olora/s1 \
  --lora_rank 8 \
  --adaptation_lr 1e-4 \
  --ortho_reg_weight 0.1 \
  --eval_period 5000 \
  --batchsize 64 \
  --model_arch seq2seq_im_mask_lora
