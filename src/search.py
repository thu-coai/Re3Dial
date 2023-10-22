import sys, os
import torch
import time
import json
import argparse
from tqdm import tqdm
import numpy as np
import datasets
from datasets import load_dataset, load_metric
from multiprocessing import Pool


glob_data_source=None 
glob_top_k=None
glob_logger=None
glob_text_column_name=None
glob_index_name=None


def get_neighbour_examples(query_item):
    query = query_item['text']
    query_id = query_item['id']
    query_embedding = query_item['embedding']
    try:
        score, hard_example = glob_data_source.get_nearest_examples(
            glob_index_name,
            query_embedding,
            k=glob_top_k
        )
        return query_id, score, hard_example['id']
    except Exception as e:
        print(e)
        if glob_logger is not None:
            glob_logger.info("error detected when search for query {}, skipped this query".format(query))
        return query_id, None, None


class UDSRSelector:

    def __init__(self, source, target, index_name, col_name="text", num_proc=4, enable_rake=False, string_factory=None, gpu=None, train_size=None):
        self.source = source

        self.target = target
        self.num_proc = num_proc
        global glob_data_source
        global glob_text_column_name
        global glob_index_name
        glob_text_column_name = col_name
        self.text_column_name = col_name
        self.index_name = index_name
        glob_index_name = index_name
        self.enable_rake = enable_rake
       
        if string_factory is not None:
            faiss_cache_path = f'{args.output_dir}/{index_name}_{string_factory}'
        else:
            faiss_cache_path = f'{args.output_dir}/{index_name}'
        if gpu is not None:
            faiss_cache_path += '_gpu'
        faiss_cache_path += '.faiss'
            
        print('faiss cache path = ', faiss_cache_path)
        if not os.path.exists(faiss_cache_path):
        # if True:
            print("index name not exist, creating new index for the training corpora")
            assert index_name in source[0]
            print('precalculate embedding! index_name = ', index_name)
            self.source.add_faiss_index(
                column=index_name, # column name
                index_name=index_name, # index name
                string_factory=string_factory,
                train_size=train_size,
                device=gpu,
            )
            glob_data_source = self.source
            self.source.save_faiss_index(index_name, faiss_cache_path)
        else:
            print('load faiss')
            self.source.load_faiss_index(
                index_name, # index name
                faiss_cache_path,
                device=gpu
            )
            glob_data_source = self.source
        print('index builded!')
        print('global index name = ', glob_index_name)

    def build_queries(self):
        print(self.target[0])
        return self.target

    def build_dataset(self, top_k, batch_size, args):
        global glob_top_k
        glob_top_k = top_k
        queries = self.build_queries()
        print('start search')

        f = open(args.raw_output_file, 'w')
        for i in tqdm(range(0, len(queries), batch_size)):
            if i + batch_size >= len(queries):
                batched_queries = queries[i:]
            else:
                batched_queries = queries[i:i+batch_size]   

            batched_query_embeddings = np.stack([i for i in batched_queries['embedding']], axis=0)
            scores, candidates = glob_data_source.get_nearest_examples_batch(
                glob_index_name,
                batched_query_embeddings,
                k=glob_top_k
            )

            for query_id, neighbour_scores, res in zip(batched_queries['id'], scores, candidates):
                neighbour_ids = res['id']
                if not any(neighbour_ids):
                    continue          
                f.write(json.dumps({"query_id": query_id, "neighbour_scores": neighbour_scores.tolist(), "neighbour_ids": neighbour_ids}))
                f.write('\n')  
            del scores
            del candidates
            del batched_queries
            del batched_query_embeddings
        print('finish search')


        f.close()
        print('save raw file')

def get_text_dataset(dataset_name):
    data_files = {}
    data_files["train"] = dataset_name
    extension = dataset_name.split(".")[-1]

    start = time.time()
    if extension == 'dataset':
        raw_datasets = datasets.Dataset.load_from_disk(dataset_name)
    else:
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files, )['train']
   
    end = time.time()
    print('load dataset, time cost = ', end - start)
    if 'embedding' in raw_datasets[0]:
        def list_to_array(data):
            return {"embedding": [np.array(vector, dtype=np.float32) for vector in data["embedding"]]} 
        raw_datasets.set_transform(list_to_array, columns='embedding', output_all_columns=True)

    print('text example = ', raw_datasets[0]['text'])   
    return raw_datasets


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--string_factory", type=str, default=None)
    parser.add_argument('--index_name', default="embeddings", type=str)
    parser.add_argument('--col_name', default="text", type=str)
    parser.add_argument('--source_file', type=str)
    parser.add_argument('--target_file', type=str)
    parser.add_argument("--output_dir", type=str)


    parser.add_argument('--raw_output_file', default="raw.json", type=str)
    parser.add_argument('--text_output_file', default="text.json", type=str)
    parser.add_argument("--text_output_len", type=int, default=None, help='convert text output len')

    parser.add_argument("--debug", action='store_true', help='debug mode')

    parser.add_argument("--gpu", nargs='+', type=int, default=None, help='gpu编号')
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--train_size", default=None, type=int)
    parser.add_argument("--num_workers", default=None, type=int)
    parser.add_argument('--top_k', default=50, type=int)
    parser.add_argument('--rake', action="store_true", help="extract key phrases in the query to speed up searching process")
    
    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.raw_output_file = os.path.join(args.output_dir, f"topk{args.top_k}"+args.raw_output_file)
    args.text_output_file = os.path.join(args.output_dir, args.text_output_file)

    for k, v in vars(args).items():
        print(f"{k}: {v}")
        
    return args


if __name__=="__main__":
    
    args = parse()
    
    torch.set_grad_enabled(False)

    source_dataset = get_text_dataset(args.source_file)
    target_dataset = get_text_dataset(args.target_file)
    
    selector = UDSRSelector(source_dataset, target_dataset, args.index_name, col_name=args.col_name, num_proc=args.num_workers, enable_rake=args.rake, string_factory=args.string_factory, gpu=args.gpu, train_size=args.train_size)

    selector.build_dataset(args.top_k, args.batch_size, args)