WORKING_DIR=/xxxx
DATA_PATH=/yyyy
SAVE_DIR=/zzz
MODEL_PATH=/KKKK

DATA_PATH=$WORKING_DIR/data/$DATASET



NUM_GPUS=$1
BATCH_SIZE=128

let GLOBAL_BATCH_SIZE=$NUM_GPUS*$BATCH_SIZE

python3 main.py \
    --train_set $DATA_PATH/train.json \
    --valid_set $DATA_PATH/val.json \
    --save_dir $SAVE_DIR \
    --lr 1e-5 \
    --max_len 256 \
    --weight_decay 0.1 \
    --warm_up 0.01 \
    --model_config $MODEL_PATH \
    --max_epochs 50 \
    --gpus $NUM_GPUS \
    --batch_size $GLOBAL_BATCH_SIZE \
    --num_neg_sample 1 \
    --gradient_checkpointing \
    --val_check_interval 1.0