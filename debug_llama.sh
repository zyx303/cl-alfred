export CUDA_VISIBLE_DEVICES=7
export ALFRED_ROOT=.
export debug=1
python models/train/train_seq2seq.py \
  --incremental_setup behavior_il_test \
  --use_clip \
  --use_llama \
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

# python models/eval/eval_seq2seq.py \
#   --model_path  /home/yongxi/work/cl-alfred/exp/behavior_il_test/olora/s1/net_epoch_000000060_pick_two_obj_and_place.pth\
#   --eval_split valid_seen \
#   --incremental_setup behavior_il \
#   --incremental_type look_at_obj_in_light \
#   --stream_seed 1 \
#   --num_threads 6 \
#   --x_display 0 \
#   --gpu \
#   --model models.model.seq2seq_im_mask_olora