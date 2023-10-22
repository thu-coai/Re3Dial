

# dialogue-level diversity sampling
SOURCE_FILE=/xxx
SEARCH_FILE=/yyy
OUT_FILE=/zzz


python diversity_sampling_dialogue_level.py \
    --query_file $QUERY_FILE \
    --source_file $SOURCE_FILE \
    --search_file $SEARCH_FILE \
    --out_file $OUT_FILE


# build corpus

QUERY_FILE=None
SOURCE_FILE=/xxx
SEARCH_FILE=/yyy
OUT_FILE=/zzz

L=5
K=5

python build_corpus.py \
    --query_file $QUERY_FILE \
    --source_file $SOURCE_FILE \
    --search_file $SEARCH_FILE \
    --out_file $OUT_FILE \
    --L $L \
    --topk $K