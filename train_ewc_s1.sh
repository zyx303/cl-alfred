export ALFRED_ROOT=~/work/cl-alfred
export CUDA_VISIBLE_DEVICES=6


python models/train/train_seq2seq.py        \
    --incremental_setup behavior_il         \
    --mode cama                             \
    --stream_seed 1                         \
    --eval_period 500 \
    --dout exp/behavior_il_new/cama/s1