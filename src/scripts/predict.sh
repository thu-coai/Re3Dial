MODEL_PATH=/yyyy
LOAD_DIR=/zzz
TEST_DATASET=/xxxx
SAVE_DIR=/kkk


NUM_GPUS=$1
BATCH_SIZE=2000
let GLOBAL_BATCH_SIZE=$NUM_GPUS*$BATCH_SIZE


for PREDICT_TYPE in context query
do
    OUTPUT_PATH=$SAVE_DIR/embedding_${PREDICT_TYPE}

    python main.py \
        --predict \
        --test_set $TEST_DATASET \
        --load_dir $LOAD_DIR/*.ckpt \
        --predict_out_path $OUTPUT_PATH \
        --gpus $NUM_GPUS \
        --max_len 256 \
        --model_config $MODEL_PATH \
        --batch_size $GLOBAL_BATCH_SIZE \
        --predict_type $PREDICT_TYPE
done