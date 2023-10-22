import argparse
from tqdm import tqdm
import os
import math
import json
import numpy as np
import pathlib
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoConfig, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataset import DialogDataset, DialogInferenceDataset

class Helper():
    def __init__(self, args):
        print('args.model_config = ', args.model_config)
        self.tokenizer = BertTokenizer.from_pretrained(args.model_config, truncation_side='left')
        self.context_tokenizer = BertTokenizer.from_pretrained(args.model_config, truncation_side='right')
        self.add_tokens()  # set all special tokens

    def add_tokens(self):
        self.utt_sep_token = '<uttsep>'
        self.tokenizer.add_tokens([self.utt_sep_token])
        self.tokenizer.add_special_tokens({"pad_token": '[PAD]'})
        self.context_tokenizer.add_tokens([self.utt_sep_token])
        self.context_tokenizer.add_special_tokens({"pad_token": '[PAD]'})

def pad_collate(batch):
    questions = [] 
    all_ctxs = []
    positive_ctx_indices = []
    ctx_mask = [] 
    for i in batch:
        context_pos = i['positive_ctx']
        context_neg = i['hard_negative_ctx']
        if len(context_neg) > args.num_neg_sample:
            context_neg = random.sample(context_neg, args.num_neg_sample)
        
        ctxs = context_pos + context_neg
        
        mask = [0] * len(ctxs)
        if len(context_neg) < args.num_neg_sample:
            # add dummy ctxs
            ctxs.extend(
                ["占位"] * (args.num_neg_sample - len(context_neg))
            )
            mask.extend([1] * (args.num_neg_sample - len(context_neg)))
        # try:
        #     assert len(ctxs) - len(context_pos) == args.num_neg_sample * len(context_pos)
        # except:
        #     print('='*100)
        #     print('pad ctxs len = ', len(ctxs))
        #     print('num neg sample = ', args.num_neg_sample)
        #     print('len context pos = ', len(context_pos))
        #     print('='*100)
        current_ctxs_len = len(all_ctxs)
        all_ctxs.extend(ctxs)
        positive_ctx_indices.append(current_ctxs_len)
        questions.append(i['query'])
        ctx_mask.extend(mask)
        
    question_tensors = helper.tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_len,
        )
    
    if args.share_tokenizer:
        ctx_tensors = helper.tokenizer(
            all_ctxs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
            )
    else:
        ctx_tensors = helper.context_tokenizer(
            all_ctxs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_len,
            )  
    
    return {
        "query_ids": dict(question_tensors),
        "context_ids": dict(ctx_tensors),
        "pos_ctx_indices": torch.tensor(positive_ctx_indices, dtype=torch.long),
        "ctx_mask": torch.tensor(ctx_mask, dtype=torch.bool),
    }
    

def pad_collate_inference(batch):
    res = {}
    res['input_ids'] = pad_sequence([torch.tensor(x['input_ids'], dtype=torch.long) for x in batch], batch_first=True,
                                    padding_value=helper.tokenizer.pad_token_id)
    res['attention_mask'] = pad_sequence([torch.tensor(x['attention_mask'], dtype=torch.float) for x in batch], batch_first=True,
                                    padding_value=0)
    res['token_type_ids'] = pad_sequence([torch.tensor(x['token_type_ids'], dtype=torch.long) for x in batch], batch_first=True,
                                    padding_value=1)
    return res


class UDSR(pl.LightningModule):
    def __init__(self, model_config=None, share_model=False, lr=None, warm_up=None, weight_decay=None, gradient_checkpointing=False, k=1, predict_type=None):
        super().__init__()

        self.lr = lr
        self.warm_up = warm_up
        self.weight_decay = weight_decay
        self.max_step = None
        self.gradient_checkpointing = gradient_checkpointing
        self.predict_type = predict_type
        
        self.k = k # k for accuracy@k metric
        self.in_batch_eval = True

        self.setup_model(model_config, share_model)

        self.loss = nn.CrossEntropyLoss()
    
    def setup_model(self, model_config, share_model):
        self.query_encoder = BertModel.from_pretrained(model_config)
        
        if share_model:
            print('share model!')
            self.context_encoder = self.query_encoder
        else:
            self.context_encoder = BertModel.from_pretrained(model_config)

        self.context_encoder.resize_token_embeddings(len(helper.tokenizer))
        self.query_encoder.resize_token_embeddings(len(helper.tokenizer))
        
        if self.gradient_checkpointing:
            self.context_encoder.gradient_checkpointing_enable()
            self.query_encoder.gradient_checkpointing_enable()
            
    def forward(self, query_ids, context_ids):
        # encode query and contexts
        outputs = self.query_encoder(**query_ids)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        query_repr = pooled_output   # bs x d
       
        outputs = self.context_encoder(**context_ids)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        context_repr = pooled_output # ctx_cnt x d
        
        return query_repr, context_repr
    
    def sim_score(self, query_repr, context_repr, mask=None):
        scores = torch.matmul(
            query_repr, torch.transpose(context_repr, 0, 1)
        )  # num_q x num_ctx
        if mask is not None:
            # mask is size num_ctx
            scores[mask.repeat(scores.size(0), 1)] = float("-inf")
        return scores

    def training_step(self, batch, batch_idx):
        query_ids = batch["query_ids"]  # bs x tokens
        context_ids = batch["context_ids"]  # ctx_cnt x ctx_len
        pos_ctx_indices = batch["pos_ctx_indices"]  # bs
        mask = batch["ctx_mask"]  # ctx_cnt
               
        query_repr, context_repr = self(query_ids, context_ids)  # bs
        return {"query_repr": query_repr, "context_repr": context_repr, "pos_ctx_indices": pos_ctx_indices, "mask": mask}
    
    def training_step_end(self, batch_parts):
        query_repr = batch_parts['query_repr']
        context_repr = batch_parts['context_repr']
        pos_ctx_indices = batch_parts['pos_ctx_indices']
        mask = batch_parts['mask']
        
        scores = self.sim_score(query_repr, context_repr, mask)
        loss = self.loss(scores, pos_ctx_indices)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def _eval_step(self, batch, batch_idx):
        query_ids = batch["query_ids"]  # bs x tokens
        context_ids = batch["context_ids"]  # ctx_cnt x ctx_len
        pos_ctx_indices = batch["pos_ctx_indices"]  # bs x ctx_cnt
        mask = batch["ctx_mask"]  # ctx_cnt
        
        query_repr, contexts_repr = self(query_ids, context_ids)
        return {
            "query_repr": query_repr,
            "context_repr": contexts_repr,
            "pos_ctx_indices": pos_ctx_indices,
            "mask": mask
        }

    def compute_rank_metrics(self, pred_scores, target_labels):
        # Compute total un_normalized avg_ranks, mrr
        values, indices = torch.sort(pred_scores, dim=1, descending=True)
        rank = 0
        mrr = 0.0
        score = 0
        for i, idx in enumerate(target_labels):
            gold_idx = torch.nonzero(indices[i] == idx, as_tuple=False)
            rank += gold_idx.item() + 1
            score += gold_idx.item() < self.k
            mrr += 1 / (gold_idx.item() + 1)
        return rank, mrr, score
    
    def _eval_step_end(self, batch_parts):
        query_repr = batch_parts['query_repr']
        context_repr = batch_parts['context_repr']
        pos_ctx_indices = batch_parts['pos_ctx_indices']
        mask = batch_parts['mask']
        
        pred_context_scores = self.sim_score(query_repr, context_repr, mask)
        return (
                self.compute_rank_metrics(pred_context_scores, pos_ctx_indices),
                query_repr,
                context_repr,
                pos_ctx_indices,
                mask
            )
    
    def _eval_epoch_end(self, outputs, log_prefix="val"):
        total_avg_rank, total_ctx_count, total_count = 0, 0, 0
        total_mrr = 0
        # total_loss = 0
        total_score = 0
        assert self.in_batch_eval
        if self.in_batch_eval:
            for metrics, query_repr, contexts_repr, _, mask in outputs:
                rank, mrr, score = metrics
                total_avg_rank += rank
                total_mrr += mrr
                total_score += score
                total_ctx_count += contexts_repr.size(0) - torch.sum(mask)
                total_count += query_repr.size(0)
                # total_loss += loss
            total_ctx_count = total_ctx_count / len(outputs)
        else:
            # collate the representation and gold +ve labels
            all_query_repr = []
            all_context_repr = []
            all_labels = []
            all_mask = []
            offset = 0
            for _, query_repr, context_repr, target_labels, mask, _ in outputs:
                all_query_repr.append(query_repr)
                all_context_repr.append(context_repr)
                all_mask.append(mask)
                all_labels.extend([offset + x for x in target_labels])
                offset += context_repr.size(0)
            # gather all contexts
            all_context_repr = torch.cat(all_context_repr, dim=0)
            all_mask = torch.cat(all_mask, dim=0)
            if self.trainer.accelerator_connector.use_ddp:
                all_context_repr, all_mask = self.all_gather(
                    (all_context_repr, all_mask)
                )
                all_labels = [
                    x + all_context_repr.size(1) * self.global_rank for x in all_labels
                ]
                all_context_repr = torch.cat(tuple(all_context_repr), dim=0)
                all_mask = torch.cat(tuple(all_mask), dim=0)
            all_query_repr = torch.cat(all_query_repr, dim=0)
            scores = self.sim_score(all_query_repr, all_context_repr, all_mask)
            total_count = all_query_repr.size(0)
            total_ctx_count = scores.size(1) - torch.sum(all_mask)
            total_avg_rank, total_mrr, total_score = self.compute_rank_metrics(
                scores, all_labels
            )

        metrics = {
            log_prefix + "_avg_rank": total_avg_rank / total_count,
            log_prefix + "_mrr": total_mrr / total_count,
            log_prefix + f"_accuracy@{self.k}": total_score / total_count,
            log_prefix + "_ctx_count": total_ctx_count,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)
    
    def validation_step_end(self, batch_parts):
        return self._eval_step_end(batch_parts)
        
    def validation_epoch_end(self, valid_outputs):
        return self._eval_epoch_end(valid_outputs)

    def predict_step(self, batch, batch_idx):
        if self.predict_type == 'context':
            outputs = self.context_encoder(**batch)
        elif self.predict_type == 'query':
            outputs = self.query_encoder(**batch)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        repr = pooled_output   # bs x d

        return repr
    
    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)
    
    def test_step_end(self, batch_parts):
        return self._eval_step_end(batch_parts)
        
    def test_epoch_end(self, test_outputs):
        return self._eval_epoch_end(test_outputs, "test")

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        betas = (0.9, 0.98)
        if self.warm_up:
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay, betas=betas)
            num_warmup_steps = int(self.max_step * self.warm_up)
            num_training_steps = self.max_step
            print("num_warmup_steps = ", num_warmup_steps)
            print("num_training_steps = ", num_training_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps) # num_warmup_steps, num_training_steps
            # scheduler = get_constant_schedule_with_warmup(optimizer, self.max_step * self.warmup)
            return (
                [optimizer],
                [
                    {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1,
                        'reduce_on_plateau': False,
                        'monitor': 'val_loss',
                    }
                ]
            )
        else:
            return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay, betas=betas)


def train(args):
    train_set = DialogDataset('train', args.train_set, args.max_len, helper.tokenizer)
    valid_set = DialogDataset('valid', args.valid_set, args.max_len, helper.tokenizer)

    train_set.show_example()
    valid_set.show_example()

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=pad_collate)
    valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  collate_fn=pad_collate)
    print('train_size = ', len(train_set))
    print('valid_size = ', len(valid_set))
    
    model = UDSR(model_config=args.model_config, lr=args.lr, warm_up=args.warm_up, weight_decay=args.weight_decay, share_model=args.share_model, gradient_checkpointing=args.gradient_checkpointing)
    model.max_step = math.ceil(len(train_set) / args.batch_size) * args.max_epochs
    # val_check_interval = min(2500, math.ceil(len(train_set) / args.batch_size))
    
    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy@1', save_top_k=1, mode='max', verbose=True, save_on_train_epoch_end=False)
    earlystop_callback = EarlyStopping(monitor='val_accuracy@1', verbose=True, mode='max')

    val_check_interval = args.val_check_interval
    trainer = pl.Trainer(gpus=int(args.gpus), max_epochs=args.max_epochs,
                         callbacks=[checkpoint_callback, earlystop_callback], val_check_interval=val_check_interval,
                         default_root_dir=args.save_dir, strategy='dp')
    trainer.fit(model, train_dataloader, valid_dataloader)
    
    with open(f"{trainer.logger.log_dir}/hyperparameter.json", 'w') as f:
        json.dump(vars(args), f, indent=2)


def test(args):
    test_set = DialogDataset('test', args.test_set, args.max_len, helper.tokenizer)
    print('test_size = ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=pad_collate)
    model = UDSR.load_from_checkpoint(args.load_dir, model_config=args.model_config, k=args.k)
    model.eval()
    trainer = pl.Trainer(gpus=int(args.gpus), logger=False, strategy='dp')
    trainer.test(model, test_dataloader, verbose=True)


def generate_embeddings(args):
    if os.path.exists(f"{args.predict_out_path}.json") or os.path.exists(f"{args.predict_out_path}.npy") or os.path.exists(f"{args.predict_out_path}.dataset"):
        print(f'find existing file: {args.predict_out_path}, continue!')
        return
        
    test_set = DialogInferenceDataset("predict", args.test_set, args.max_len, helper.tokenizer)
    print('test_size = ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=pad_collate_inference)
    model = UDSR.load_from_checkpoint(args.load_dir, model_config=args.model_config, predict_type=args.predict_type)
    model.eval()

    trainer = pl.Trainer(gpus=int(args.gpus), logger=False, strategy='dp') # disable logging
    embeds = trainer.predict(model, test_dataloader)

    embeds = torch.cat(embeds, dim=0)
    embeds= embeds.cpu().numpy()
    
    test_set.dump(embeds, args.predict_out_path)


def parse():
    parser = argparse.ArgumentParser(description='finetune bert')

    # data path
    parser.add_argument('--train_set', type=str, 
                        help='Path of training set')
    parser.add_argument('--valid_set', type=str,
                        help='Path of validation set')
    parser.add_argument('--test_set', type=str,
                        help='Path of test set')

    # device
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of cpu cores to use')
    parser.add_argument('--gpus', default=None, help='Gpus to use')

    # hyperparameters
    parser.add_argument("--share_model", type=int, default=0,
                        help='是否share question encoder和context encoder')
    parser.add_argument("--share_tokenizer", type=int, default=1)
    parser.add_argument("--num_neg_sample", type=int, default=1,)
    
    parser.add_argument("--val_check_interval", type=float, default=1.0, help='val check interval')
    
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument("--seed", type=int, default=42, 
                        help='random seed')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="weight_decay")
    parser.add_argument("--warm_up", type=float, default=None,
                        help="warm up rate")
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of mini-batch')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle data')
    parser.add_argument("--num_labels", type=int, default=2,
                        help="classification num labels")
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--k", type=int, default=1, help='accuracy@k')

    # model type
    parser.add_argument("--model_config", type=str,
                        help="config path for model(tokenizer)")

    # model load/save
    parser.add_argument('--load_dir', type=str, default=None,
                        help='Directory of checkpoint to load for predicting')
    parser.add_argument('--save_dir', type=str,
                        help='Path to save model')
    parser.add_argument('--predict_out_path', type=str, default=None,
                    help='Path of prediction file')

    # mode
    parser.add_argument('--predict', action='store_true',
                    help='predict result')
    parser.add_argument("--predict_type", type=str, default=None, 
                    help='[context, query]')
    parser.add_argument('--test', action='store_true',
                help='test')
    
    args = parser.parse_args()
    
    
    if args.predict_out_path is not None:
        directory = pathlib.Path(args.predict_out_path).parents[0]
        if not os.path.exists(directory):
            os.makedirs(directory)
        
    return args


if __name__ == '__main__':

    args = parse()
    for k, v in vars(args).items():
        print(f"{k}:\t{v}")
    helper = Helper(args)

    pl.seed_everything(args.seed)

    if args.predict:
        generate_embeddings(args)
    elif args.test:
        test(args)
    else:
        train(args)