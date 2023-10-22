SOURCE_FILE=/xxx
TARGET_FILE=/yyy
OUTPUT_DIR=/zzz

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$GPUID python search.py \
    --index_name embedding \
    --col_name text \
    --source_file $SOURCE_FILE \
    --target_file $TARGET_FILE \
    --output_dir $OUTPUT_DIR \
    --batch_size 32 \
    --gpu 0 \
    --top_k 10
    