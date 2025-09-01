export ALFRED_ROOT=~/work/cl-alfred
export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0
export debug=1

python models/eval/eval_seq2seq.py                                                    \
    --model_path /home/yongxi/work/cl-alfred/exp/behavior_il_new/olora_12/s1/net_epoch_000062328_pick_heat_then_place_in_recep.pth \
    --eval_split valid_seen                                                           \
    --incremental_setup behavior_il                                                   \
    --incremental_type pick_heat_then_place_in_recep                                           \
    --stream_seed 1                                                                   \
    --num_threads 1                                                                    \
    --x_display 0                                                                       \
    --gpu