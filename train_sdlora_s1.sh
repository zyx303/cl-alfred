#!/bin/bash

# SD-LoRA training script for CL-ALFRED

export CUDA_VISIBLE_DEVICES=7
export ALFRED_ROOT=~/work/cl-alfred

python models/train/train_seq2seq.py \
  --data data/json_feat_2.1.0 \
  --model_arch seq2seq_im_mask \
  --dout exp/behavior_il/sdlora/s1 \
  --splits data/splits/oct21.json \
  --preprocess \
  --pp_folder pp \
  --gpu \
  --batch 32 \
  --pm_aux_loss_wt 0.1 \
  --subgoal_aux_loss_wt 0.1 \
  --mode sdlora \
  --n_tasks 7 \
  --memory_size 500 \
  --lr 1e-3 \
  --lora_rank 10 \
  --lora_alpha 1.0 \
  --adaptation_lr 1e-4 \
  --ortho_reg_weight 0.1 \
  --incremental_setup behavior_il \
  --stream_seed 1 \
  --eval_period 500 \
  --temp_batchsize 16
