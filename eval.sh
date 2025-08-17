export ALFRED_ROOT=~/work/cl-alfred
export CUDA_VISIBLE_DEVICES=6,7

python models/eval/eval_seq2seq.py                                                    \
    --model_path exp/behavior_il/cama/s1/net_epoch_000002251_look_at_obj_in_light.pth \
    --eval_split valid_seen                                                           \
    --incremental_setup behavior_il                                                   \
    --incremental_type look_at_obj_in_light                                           \
    --stream_seed 1                                                                   \
    --num_threads 2                                                                   \
    --x_display 1                                                                     \
    --gpu