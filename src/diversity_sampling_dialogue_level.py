
from operator import pos
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
import time



def overlap_filter(query, candidate_texts, candidate_ids):
    res_ids = [] 
    res_filter_infos = []

    bound = 10
    
    sep_token = None
    if '\t\t' in query:
        sep_token = '\t\t'
    elif '\t' in query:
        sep_token = '\t'
    # print('sep_token = ', sep_token)
    assert sep_token is not None
    query_utts = query.strip().split(sep_token)

    # query_md5 = hashlib.md5(query.encode("utf-8")).digest()
    for session, session_id in zip(candidate_texts, candidate_ids):
        filter_info = ''
        utts = session.strip().split(sep_token)
        flag = False

        for utt in utts:
            for i in query_utts:
                if i == utt:
                    flag = True
                    filter_info = 'exact_match'
                    break
                elif (utt in i or i in utt):
                    flag = True
                    filter_info = 'contain'
                    break
                else:
                    overlap_score = pylcs.lcs2(i, utt)
                    if overlap_score > bound:
                        flag = True
                        filter_info= f'lcs: {overlap_score}, bound: {bound}'
                        break
            if flag:
                break
            
        res_filter_infos.append(filter_info)
        if not flag:
            res_ids.append(session_id)
    return res_ids, res_filter_infos


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


    cnt = 0

    for idx, query_id in tqdm(enumerate(search_ids)):
        candidates = search_ids[query_id]['neighbour_ids']
        ori_num = len(candidates)
        
        query_text = source_list[query_id]
        
        candidates_text = [source_list[i] for i in candidates]
        candidates, filter_infos = overlap_filter(query_text, candidates_text, candidates)

        search_ids[query_id]['neighbour_ids'] = candidates

        filtered_num = len(candidates)

        cnt += ori_num - filtered_num

    print('filter cnt = ', cnt)

    start = time.time()
    with open(args.out_file, 'w') as f:
        for item in tqdm(search_ids.values()):
            f.write(json.dumps(item, ensure_ascii=False))
            f.write('\n')
    end = time.time()
    print('raw id saved, time cost = ', end - start)
    

def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source_file", type=str, default=None)
    parser.add_argument("--query_file", type=str, default=None)
    parser.add_argument("--search_file", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--filter_file", type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    random.seed(42)
    args = parse()
    main(args)


