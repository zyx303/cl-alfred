export CUDA_VISIBLE_DEVICES=7
export ALFRED_ROOT=.
CKPT_PATH=/home/yongxi/work/cl-alfred/exp/behavior_il_new/olora/s1/net_epoch_000022510_look_at_obj_in_light.pth

python eval_param.py \
  --resume ${CKPT_PATH} \
  --incremental_setup behavior_il_test \
  --mode olora \
  --stream_seed 1 \
  --dout exp/behavior_il_test/olora/s1 \
  --lora_rank 8 \
  --adaptation_lr 1e-3 \
  --lamda_1 0.5 \
  --lamda_2 0.0 \
  --eval_period 5000 \
  --batchsize 64 \
  --epochs_per_task 1\
  --model_arch seq2seq_im_mask_olora


# CKPT_PATH=/home/yongxi/work/cl-alfred/exp/behavior_il_new/olora/s1/net_epoch_000051940_pick_heat_then_place_in_recep.pth

# python eval_param.py \
#   --resume ${CKPT_PATH} \
#   --incremental_setup behavior_il_test \
#   --mode olora \
#   --stream_seed 1 \
#   --dout exp/behavior_il_test/olora/s1 \
#   --lora_rank 8 \
#   --adaptation_lr 1e-3 \
#   --lamda_1 0.5 \
#   --lamda_2 0.0 \
#   --eval_period 5000 \
#   --batchsize 64 \
#   --epochs_per_task 1\
#   --model_arch seq2seq_im_mask_olora


# CKPT_PATH=exp/behavior_il/er/s1/net_epoch_000002251_look_at_obj_in_light.pth

# python eval_param.py \
#     --resume ${CKPT_PATH} \
#     --incremental_setup behavior_il_test \
#     --mode olora \
#     --stream_seed 1 \
#     --dout exp/behavior_il_test/olora/s1 \
#     --lora_rank 8 \
#     --adaptation_lr 1e-3 \
#     --lamda_1 0.5 \
#     --lamda_2 0.0 \
#     --eval_period 5000 \
#     --batchsize 64 \
#     --epochs_per_task 1\
#     --model_arch seq2seq_im_mask