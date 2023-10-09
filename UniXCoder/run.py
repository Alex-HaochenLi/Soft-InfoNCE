# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import sys 
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
import pandas as pd
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                              RobertaConfig, RobertaModel, RobertaTokenizer, AutoModel, AutoTokenizer)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,
                 q_ids

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.q_ids = q_ids

        
def convert_examples_to_features(js,tokenizer,args,nl_tokenizer):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    q_ids = None
    if nl_tokenizer is not None:
        q = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
        q_tokens = nl_tokenizer.tokenize(q)[:args.nl_length - 2]
        q_tokens = [nl_tokenizer.cls_token] + q_tokens + [nl_tokenizer.sep_token]
        q_ids = nl_tokenizer.convert_tokens_to_ids(q_tokens)
        padding_length = args.nl_length - len(q_ids)
        q_ids += [nl_tokenizer.pad_token_id] * padding_length
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"],q_ids)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, nl_tokenizer=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js) 

        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,args,nl_tokenizer))
                
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

        total_len = 0
        for i in range(len(self.examples)):
            total_len += len(self.examples[i].nl_tokens)
        self.avg_len = total_len / len(self.examples)
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.examples[i].q_ids is None:
            return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))
        else:
            return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids),torch.tensor(self.examples[i].q_ids))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

k1 = 1.2
k2 = 100
b = 0.75
R = 0.0
from math import log


def score_BM25(n, f, qf, r, N, dl, avdl):
    K = compute_K(dl, avdl)
    first = log( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2+1) * qf) / (k2 + qf)
    return first * second * third


def compute_K(dl, avdl):
    return k1 * ((1-b) + b * (float(dl)/float(avdl)) )


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def train(args, model, tokenizer, model_predict=None, nl_tokenizer=None):
    """ Train the model """
    #get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file, nl_tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * args.num_train_epochs)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*args.num_train_epochs)
    
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    
    model.train()
    if args.simcse or args.self:
        model_predict.eval()
    tr_num,tr_loss,best_mrr = 0,0,0 
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(args.device)    
            nl_inputs = batch[1].to(args.device)
            if nl_tokenizer is not None:
                q_inputs = batch[2].to(args.device)
            labels = torch.ones(nl_inputs.size(0), device=args.device).long()
            loss_mask = labels.diag()

            if args.bm25:
                dict1 = dict()
                for docidx, tokens in enumerate(nl_inputs):
                    for i in tokens:
                        if i > 3:
                            if i.item() in dict1:
                                if docidx in dict1[i.item()]:
                                    dict1[i.item()][docidx] += 1
                                else:
                                    dict1[i.item()][docidx] = 1
                            else:
                                d = dict()
                                d[docidx] = 1
                                dict1[i.item()] = d
                avg_len = train_dataset.avg_len
                sim_matrix = torch.zeros((nl_inputs.size(0), nl_inputs.size(0)), device=args.device)
                for i, query in enumerate(nl_inputs):
                    for token in query:
                        if token > 3:
                            doc_dict = dict1[token.item()]  # retrieve index entry
                            for docid, freq in doc_dict.items():  # for each document and its word frequency
                                score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=nl_inputs.size(0),
                                                   dl=(nl_inputs[docid] > 3).sum().item(),
                                                   avdl=avg_len)  # calculate score
                                sim_matrix[i][docid] += score

            if args.self:
                with torch.no_grad():
                    code_vec = model_predict(code_inputs=code_inputs)
                    nl_vec = model_predict(nl_inputs=nl_inputs)
                    sim_matrix = torch.matmul(nl_vec, code_vec.T)

            if args.simcse:
                with torch.no_grad():
                    queries = model_predict(q_inputs, return_dict=True)
                    queries = mean_pooling(queries, q_inputs.ne(0))
                    queries = torch.nn.functional.normalize(queries, dim=1, p=2)
                    sim_matrix = torch.matmul(queries, queries.T)

            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)
            
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)

            sim_neg = sim_matrix[loss_mask != 1].view(sim_matrix.size(0), -1)

            if args.softinfonce:
                weights = torch.nn.functional.softmax(sim_neg / args.t, dim=1)
                alpha, beta = args.weight_kl, args.weight_unif
                weights = torch.clip((beta - alpha * weights) / (beta - alpha / weights.size(1)), min=0.1)
                weights = torch.cat([torch.ones((weights.size(0), 1), device=weights.device), weights], dim=1)
                scores = torch.cat([scores[loss_mask == 1].unsqueeze(1), scores[loss_mask != 1].view(scores.size(0), -1)],
                                   dim=1)

                maxes = torch.max(scores, 1, keepdim=True)[0]
                x_exp = torch.exp(scores - maxes)
                x_exp_sum = torch.sum(weights * x_exp, 1, keepdim=True)
                probs = x_exp / x_exp_sum
                loss = - torch.mean(torch.log(probs[:, 0] + 1e-15))

            if args.weightedinfonce:
                weights = torch.nn.functional.softmax(sim_neg / args.t, dim=1)
                weights = torch.cat([torch.ones((weights.size(0), 1), device=weights.device), weights], dim=1)
                scores = torch.cat(
                    [scores[loss_mask == 1].unsqueeze(1), scores[loss_mask != 1].view(scores.size(0), -1)],
                    dim=1)
                scores = torch.nn.functional.softmax(scores, dim=1)
                loss = - torch.mean((torch.log(scores + 1e-15) * weights).sum(1))

            if args.infonce:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(scores, torch.arange(code_inputs.size(0), device=scores.device))

            if args.bce:
                weights = torch.nn.functional.softmax(sim_neg / args.t, dim=1)
                weights = torch.cat([torch.ones((weights.size(0), 1), device=weights.device), weights], dim=1)
                scores = torch.cat([scores[loss_mask == 1].unsqueeze(1), scores[loss_mask != 1].view(scores.size(0), -1)],
                                   dim=1)
                loss_fct = torch.nn.BCELoss()
                scores = torch.nn.functional.softmax(scores, dim=1)
                loss = loss_fct(scores, weights)

            if args.klregularization:
                weights = torch.nn.functional.softmax(sim_neg / args.t, dim=1)
                loss_fct2 = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
                loss2 = loss_fct2(torch.nn.functional.log_softmax(scores[loss_mask != 1].view(scores.size(0), -1), dim=1), weights.log())
                scores = torch.cat([scores[loss_mask == 1].unsqueeze(1), scores[loss_mask != 1].view(scores.size(0), -1)],
                                   dim=1)
                loss_fct1 = CrossEntropyLoss()
                loss1 = loss_fct1(scores, torch.zeros(code_inputs.size(0), device=scores.device).long())
                loss = args.weight_unif * loss1 + args.weight_kl * loss2
            
            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%(len(train_dataloader) // 5) == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 
            
        #evaluate    
        results = evaluate(args, model, tokenizer,args.eval_data_file, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))

        # save epoch model
        checkpoint_prefix = 'checkpoint-last'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_dir = os.path.join(output_dir, '{}'.format('pytorch_model.bin'))
        torch.save(model_to_save.state_dict(), output_dir)

        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s",round(best_mrr,4))
            logger.info("  "+"*"*20)                          

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)                        
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('pytorch_model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer,file_name,eval_when_training=False):
    query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size,num_workers=4)
    
    code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size,num_workers=4)    

    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    
    model.eval()
    code_vecs = [] 
    nl_vecs = []
    for batch in query_dataloader:  
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)    
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())  
    model.train()    
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    scores = np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls,sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr":float(np.mean(ranks))
    }

    return result

                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  
    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--lang', type=str, default=None,
                        help="language for retrieval on CSN Challenge")
    parser.add_argument('--t', type=float, default=1,
                        help="temperature for sharpening weights")
    parser.add_argument("--weight_kl", default=1, type=float,
                        help='temperature for sharpening lambda')
    parser.add_argument("--weight_unif", default=1, type=float,
                        help='temperature for sharpening lambda')
    parser.add_argument("--predict_checkpoint_path", default=None, type=str,
                        help='model for calculating similarity')
    parser.add_argument("--bm25", action='store_true',
                        help="Use BM25 for weight estimation.")
    parser.add_argument("--self", action='store_true',
                        help="Use trained model for weight estimation.")
    parser.add_argument("--simcse", action='store_true',
                        help="Use unsupervised SimCSE model for weight estimation.")
    parser.add_argument("--infonce", action='store_true',
                        help="Use InfoNCE as loss function.")
    parser.add_argument("--softinfonce", action='store_true',
                        help="Use Soft-InfoNCE as loss function.")
    parser.add_argument("--bce", action='store_true',
                        help="Use Binary Cross Entropy as loss function.")
    parser.add_argument("--weightedinfonce", action='store_true',
                        help="Use weighted InfoNCE as loss function.")
    parser.add_argument("--klregularization", action='store_true',
                        help="Use InfoNCE + KL Div regularization as loss function.")

    #print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)

    nl_tokenizer = None
    model_predict = None
    if args.self:
        model_predict = Model(model)
        model_predict.load_state_dict(torch.load(os.path.join(args.predict_checkpoint_path, 'pytorch_model.bin')))

    if args.simcse:
        nl_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-dot-v5")
        model_predict = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-dot-v5")
        model_predict.load_state_dict(torch.load('./saved_models/simcse-cos0.07/checkpoint-best-mrr/pytorch_model.bin'))

    model_train = Model(model)
    logger.info("Training/evaluation parameters %s", args)
    
    model_train.to(args.device)
    if args.simcse or args.self:
        model_predict.to(args.device)
    if args.n_gpu > 1:
        model_train = torch.nn.DataParallel(model_train)

    # Training
    if args.do_train:
        if args.simcse or args.self:
            train(args, model_train, tokenizer, model_predict, nl_tokenizer)
        else:
            train(args, model_train, tokenizer)
      
    # Evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'pytorch_model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model_train.module if hasattr(model, 'module') else model_train
            model_to_load.load_state_dict(torch.load(output_dir))      
        model_train.to(args.device)
        result = evaluate(args, model_train, tokenizer, args.eval_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key],3)))
            
    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'pytorch_model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model_to_load = model_train.module if hasattr(model, 'module') else model_train
            model_to_load.load_state_dict(torch.load(output_dir))      
        model_train.to(args.device)
        result = evaluate(args, model_train, tokenizer, args.test_data_file)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 3)))


if __name__ == "__main__":
    main()


