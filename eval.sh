export ALFRED_ROOT=~/work/cl-alfred
export CUDA_VISIBLE_DEVICES=0
export DISPLAY=:0

python models/eval/eval_seq2seq.py                                                    \
    --model_path exp/behavior_il/cama/s1/net_epoch_000002251_look_at_obj_in_light.pth \
    --eval_split valid_seen                                                           \
    --incremental_setup behavior_il                                                   \
    --incremental_type look_at_obj_in_light                                           \
    --stream_seed 1                                                                   \
    --num_threads 6                                                                    \
    --x_display 0                                                                       \
    --gpu