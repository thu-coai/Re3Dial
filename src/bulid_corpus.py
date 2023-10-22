from datasets import load_dataset
import json
from search import get_text_dataset
import random
from tqdm import tqdm
import argparse
import difflib
import pylcs
import numpy as np
from collections import Counter
import hashlib




def main(args):

    source_dataset = get_text_dataset(args.source_file)
    source_list = source_dataset['text']
    print(source_list[0])
    del source_dataset


    search_ids = {}
    with open(args.search_file, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            search_ids[int(item['query_id'])] = {"query_id": item['query_id'], "neighbour_ids": item['neighbour_ids']}

    print(f'source size = {len(source_list)}, search size = {len(search_ids)}')

    res = []

    print('missing ids = ', len(set(range(len(source_list))) - set([int(i) for i in search_ids.keys()])))

    used_session = Counter()

    for root_id in tqdm(search_ids):
        cur_path = [root_id]
        cur_set = set([root_id])

        cur_id = root_id
        used_session[cur_id] += 1 
        for hop in range(args.L - 1):
            if cur_id not in search_ids:
                break

            topk_ids = search_ids[cur_id]['neighbour_ids']
            if args.topk is not None:
                topk_ids = topk_ids[: args.topk]
            topk_ids =  set(topk_ids)
            candidates = list(topk_ids - cur_set)
            if not any(candidates): 
                break
            else:
                weights = [1 / (1 + used_session[i]) for i in candidates] # corpus-level weight
                next_id = random.choices(candidates, k=1, weights=weights)[0]
                
                cur_id = next_id
                cur_path.append(cur_id)
                cur_set |= set(topk_ids)
                used_session[cur_id] += 1

        res.append(cur_path)

    print('finish')

    text_res = []
    for path in res:
        tmp = []
        for i, idx in enumerate(path):
            text = source_list[idx]
            tmp.append(
                {
                    "text": text,
                    "id": idx
                }
            )
        text_res.append(tmp)

    print('output size = ', len(text_res))

    with open(args.out_file, 'w') as f:
        json.dump(text_res, f, indent=2, ensure_ascii=False)


def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source_file", type=str, default=None)
    parser.add_argument("--query_file", type=str, default=None)
    parser.add_argument("--search_file", type=str)
    parser.add_argument("--out_file", type=str)

    parser.add_argument("--topk", type=int, default=None)

    parser.add_argument("--L", type=int, default=4)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    random.seed(42)
    args = parse()
    main(args)


