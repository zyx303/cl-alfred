export ALFRED_ROOT=~/work/cl-alfred
export CUDA_VISIBLE_DEVICES=7


python models/train/train_seq2seq.py        \
    --incremental_setup behavior_il         \
    --mode er                             \
    --stream_seed 1                         \
    --eval_period 5000 \
    --dout exp/behavior_il_new/er/s1