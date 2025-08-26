#!/bin/bash

# SD-LoRA training script for CL-ALFRED

export CUDA_VISIBLE_DEVICES=6
export ALFRED_ROOT=.

python models/train/train_seq2seq.py \
  --incremental_setup behavior_il \
  --mode sdlora \
  --stream_seed 1 \
  --dout exp/behavior_il/sdlora/s1 \
  --lora_rank 10 \
  --lora_alpha 1.0 \
  --adaptation_lr 1e-4 \
  --ortho_reg_weight 0.1 \
  --eval_period 5000 \
  --batchsize 128 \
  --model_arch seq2seq_im_mask_lora
