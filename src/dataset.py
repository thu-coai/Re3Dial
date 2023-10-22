import json
import jsonlines
import random
import csv
from tqdm import tqdm
import time
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset
import datasets

def get_text_dataset(dataset_name):
    data_files = {}
    data_files["train"] = dataset_name
    extension = dataset_name.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files)
    return raw_datasets["train"]

class DialogDataset(Dataset):
    def __init__(self, split, path, max_len, tokenizer):
        super().__init__()

        self.split = split
        self.max_len = max_len
        self.tokenizer= tokenizer
        print('load split = ', split)
        self.data = self.load_data(path)
    
    def load_data(self, path):
       
        def process(text):
            return '<uttsep>'.join(text.split('\t\t'))
       
        if "dataset" in path: # huggingface datasets:
            ori_data = get_text_dataset(path)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                ori_data = json.load(f)
            
        data = []
        for i in tqdm(ori_data):
            item = {}
            item['query'] = process(i['query'])
            item['positive_ctx'] = [process(j) for j in i['positive_ctx']]
            item['hard_negative_ctx'] = [process(j) for j in i['hard_negative_ctx']]
            data.append(item)
            
        return data
        

    def __len__(self):
        return len(self.data)

    def show_example(self):
        print('data example:')
        print(self.data[0])
        tmp = self.__getitem__(0)
        print(tmp)

    def __getitem__(self, idx):
        ori_item = self.data[idx]
        return ori_item
    

def get_text_dataset(dataset_name):
    data_files = {}
    data_files["train"] = dataset_name
    extension = dataset_name.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files)['train']
    return raw_datasets
    
class DialogInferenceDataset(Dataset):
    def __init__(self, split, path, max_len, tokenizer):
        super().__init__()

        self.data = self.load_data(path)
        self.split = split
        self.max_len = max_len
        self.tokenizer= tokenizer
        
        self.show_example()
    
    def load_data(self, path):
        ori_data = get_text_dataset(path)
        print('load ori data, size = ', len(ori_data))    
        
        start = time.time()
        def list_to_array(data):
            return {"text": ['<uttsep>'.join(text.strip().split('\t')) for text in data["text"]]} 
        ori_data.set_transform(list_to_array, columns='text', output_all_columns=True)
        end = time.time()
        print('transform data, time cost = ', end - start)

        print(ori_data[0])
        return ori_data
        

    def __len__(self):
        return len(self.data)

    def show_example(self):
        print('data example:')
        print(self.data[0])
        tmp = self.__getitem__(0)
        print(tmp)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        return self.tokenizer(text, max_length=self.max_len, truncation=True)
    
    def dump(self, embeddings, output_path):
        assert len(self.data) == len(embeddings)
        
        start = time.time()
        dataset_embed = datasets.Dataset.from_dict({'embedding': embeddings, "id": list(range(len(self.data)))})

        end = time.time()
        print('time cost = ', end - start)
        
        final_dataset = datasets.concatenate_datasets([self.data, dataset_embed], axis=1)
        final_dataset.save_to_disk(f'{output_path}.dataset')