from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import json
import re
import shutil
from more_itertools import chunked

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from models import ModelBinary, ModelContra
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          AutoModel, AutoTokenizer)

logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, label, idx, code_len, query_len, url, q_ids):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.label = label
        self.idx = idx
        self.code_len = code_len
        self.query_len = query_len
        self.url = url
        self.q_ids = q_ids


class InputFeaturesTrip(InputFeatures):
    """A single training/test features for a example. Add docstring seperately. """
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, ds_tokens, ds_ids, label, idx):
        super(InputFeaturesTrip, self).__init__(code_tokens, code_ids, nl_tokens, nl_ids, label, idx)
        self.ds_tokens = ds_tokens
        self.ds_ids = ds_ids


def convert_examples_to_features(js, tokenizer, args, len_list, nl_tokenizer=None):
    # label
    label = js['label']

    # code
    code = js['code']
    if args.code_type == 'code_tokens':
        code = js['code_tokens']
    code_tokens = tokenizer.tokenize(code)[:args.max_seq_length-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.max_seq_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    code_len = len(code_tokens)
    len_list.append(code_len)

    nl = js['doc']
    nl_tokens = tokenizer.tokenize(nl)[:args.max_seq_length-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.max_seq_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length
    query_len = len(nl_tokens)

    q_ids = None
    if nl_tokenizer is not None:
        q = js['doc']
        q_tokens = nl_tokenizer.tokenize(q)[:args.max_seq_length - 2]
        q_tokens = [nl_tokenizer.cls_token] + q_tokens + [nl_tokenizer.sep_token]
        q_ids = nl_tokenizer.convert_tokens_to_ids(q_tokens)
        padding_length = args.max_seq_length - len(q_ids)
        q_ids += [nl_tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, label, js['idx'], code_len, query_len, js['url'], q_ids)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, nl_tokenizer=None):
        self.examples = []
        self.data = []
        self.lens = []
        self.nl_tokenizer = nl_tokenizer
        with open(file_path, 'r') as f:
            self.data = f.readlines()
        if args.debug:
            self.data = self.data[:args.n_debug_samples]
        for idx, js in enumerate(self.data):
            js = json.loads(js.strip())
            js = {'code': ' '.join(js['code_tokens']), 'doc': ' '.join(js['docstring_tokens']), 'label': 1, 'idx': idx, 'url': js['url']}

            self.examples.append(convert_examples_to_features(js, tokenizer, args, self.lens, nl_tokenizer))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:1]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

        total_len = 0
        for i in range(len(self.examples)):
            total_len += len(self.examples[i].nl_tokens)
        self.avg_len = total_len / len(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        if self.nl_tokenizer is not None:
            return torch.tensor(self.examples[i].code_ids), \
                   torch.tensor(self.examples[i].nl_ids),\
                   torch.tensor(int(self.examples[i].label)), \
                   torch.tensor(self.examples[i].code_len), \
                   torch.tensor(self.examples[i].query_len), \
                   torch.tensor(self.examples[i].q_ids)
        else:
            return torch.tensor(self.examples[i].code_ids), \
                   torch.tensor(self.examples[i].nl_ids), \
                   torch.tensor(int(self.examples[i].label)), \
                   torch.tensor(self.examples[i].code_len), \
                   torch.tensor(self.examples[i].query_len)


class RetrievalDatasetForCleanCSN(Dataset):
    def __init__(self, tokenizer, args, data_path=None):
        self.examples = []
        self.lens = []

        with open(data_path, 'r') as f:
            self.data = f.readlines()
        for idx, js in enumerate(self.data):
            js = json.loads(js.strip())
            new_js = {'code': ' '.join(js['code_tokens']), 'doc': ' '.join(js['docstring_tokens']), 'label': 0, 'idx': idx, 'url': js['url']}
            self.examples.append(convert_examples_to_features(new_js, tokenizer, args, self.lens))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return torch.tensor(self.examples[i].code_ids), \
               torch.tensor(self.examples[i].nl_ids), \
               torch.tensor(self.examples[i].label), \
               torch.tensor(self.examples[i].code_len), \
               torch.tensor(self.examples[i].query_len)


def set_seed(seed=45):
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


def train(args, train_dataset, model, tokenizer, model_predict):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1, pin_memory=True)

    args.save_steps = len(train_dataloader) if args.save_steps<=0 else args.save_steps
    args.warmup_steps = len(train_dataloader) if args.warmup_steps<=0 else args.warmup_steps
    args.logging_steps = len(train_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps)
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    model.to(args.device)
    if args.simcse or args.self:
        model_predict.eval()
        model_predict.to(args.device)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_results = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "acc_and_f1": 0.0, "mrr": 0.0}
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    # logger.info(model) #to uncomment

    for idx in tqdm(range(args.start_epoch, int(args.num_train_epochs))):

        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)
            # ds_inputs = batch[2].to(args.device)
            labels = batch[2].to(args.device)
            code_lens = batch[3].to(args.device)
            if train_dataset.nl_tokenizer is not None:
                q_inputs = batch[5].to(args.device)

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


            if args.simcse:
                with torch.no_grad():
                    queries = model_predict(q_inputs, return_dict=True)
                    queries = mean_pooling(queries, q_inputs.ne(0))
                    queries = torch.nn.functional.normalize(queries, dim=1, p=2)
                    sim_matrix = torch.matmul(queries, queries.T)

            if args.self:
                with torch.no_grad():
                    code_vec, nl_vec = model_predict(code_inputs, nl_inputs, labels, code_lens, return_vec=True)
                    sim_matrix = torch.matmul(nl_vec, code_vec.T)

            model.train()
            loss, predictions = model(code_inputs, nl_inputs, labels, code_lens, sim_matrix)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)

            if (global_step + 1) % (len(train_dataloader) // 5) == 0:
                logger.info("epoch {} step {} loss {}".format(idx, step+1, avg_loss))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        mrr = retrieval_CSN(args, model, tokenizer)
                        logger.info(" Mrr = %s", round(mrr, 4))

                        # Save model checkpoint
                        if mrr >= best_results['mrr']:
                            best_results['mrr'] = mrr

                            # save
                            checkpoint_prefix = 'checkpoint-best-mrr'
                            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model

                            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                            logger.info("Saving model checkpoint to %s", output_dir)

                    if args.local_rank == -1:
                        checkpoint_prefix = 'checkpoint-last'
                        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                        tokenizer.save_pretrained(output_dir)

                        idx_file = os.path.join(output_dir, 'idx_file.txt')
                        with open(idx_file, 'w', encoding='utf-8') as idxf:
                            idxf.write(str(args.start_epoch + idx) + '\n')
                        step_file = os.path.join(output_dir, 'step_file.txt')
                        with open(step_file, 'w', encoding='utf-8') as stepf:
                            stepf.write(str(global_step) + '\n')


def retrieval_CSN(args, model, tokenizer):
    if not args.retrieval_predictions_output:
        args.retrieval_predictions_output = os.path.join(args.output_dir, 'retrieval_outputs.txt')
    if not os.path.exists(os.path.dirname(args.retrieval_predictions_output)):
        os.makedirs(os.path.dirname(args.retrieval_predictions_output))

    q_retrieval_dataset = RetrievalDatasetForCleanCSN(tokenizer, args, '{}/{}/test.jsonl'.format(args.data_dir, args.lang))
    args.eval_batch_size = args.per_gpu_retrieval_batch_size * max(1, args.n_gpu)
    q_eval_sampler = SequentialSampler(q_retrieval_dataset)
    q_eval_dataloader = DataLoader(q_retrieval_dataset, sampler=q_eval_sampler, batch_size=args.eval_batch_size)

    c_retrieval_dataset = RetrievalDatasetForCleanCSN(tokenizer, args, '{}/{}/codebase.jsonl'.format(args.data_dir, args.lang))
    args.eval_batch_size = args.per_gpu_retrieval_batch_size * max(1, args.n_gpu)
    c_eval_sampler = SequentialSampler(c_retrieval_dataset)
    c_eval_dataloader = DataLoader(c_retrieval_dataset, sampler=c_eval_sampler, batch_size=args.eval_batch_size)


    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num queries = %d", len(q_retrieval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_first_vec = []
    all_second_vec = []
    all_code_lens, all_query_lens = [], []
    for batch in q_eval_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        query_lens = batch[4]
        all_query_lens.append(query_lens)
        with torch.no_grad():
            code_vec, nl_vec = model(code_inputs, nl_inputs, None, None, return_vec=True)
            all_second_vec.append(nl_vec.cpu())

    for batch in c_eval_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        code_lens = batch[3]
        all_code_lens.append(code_lens)
        with torch.no_grad():
            code_vec, nl_vec = model(code_inputs, nl_inputs, None, None, return_vec=True)
            all_first_vec.append(code_vec.cpu())

    batched_code = torch.cat(all_first_vec, 0)
    batched_query = torch.cat(all_second_vec, 0)

    results = []
    mrr = 0
    with torch.no_grad():
        scores = torch.matmul(batched_query, batched_code.T).numpy()

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = []
    code_urls = []
    for example in q_retrieval_dataset.examples:
        nl_urls.append(example.url)

    for example in c_retrieval_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    Recall10, Recall50 = [], []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
            if rank <= 10:
                Recall10.append(1)
                Recall50.append(1)
            elif rank > 10 and rank <= 50:
                Recall10.append(0)
                Recall50.append(1)
            else:
                Recall10.append(0)
                Recall50.append(0)
        else:
            ranks.append(0)
            Recall10.append(0)
            Recall50.append(0)

        mrr = float(np.mean(ranks))
        r10 = float(np.mean(Recall10))
        r50 = float(np.mean(Recall50))
        logger.info("  Final test MRR {} R@10 {} R@50 {}".format(mrr, r10, r50))
        return mrr


def parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--retrieval_code_base", default=None, type=str,
                        help="An optional input retrieval evaluation data file for all code bases.")

    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="The checkpoint path of model to continue training.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as encoder_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as encoder_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--debug", action='store_true',
                        help='debug mode')
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_retrieval", action='store_true',
                        help="Whether to run test on the dev/test set and do code retrieval.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--per_gpu_retrieval_batch_size", default=67, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--mrr_rank", default=100, type=int,
                        help="Number of retrieved item for computing mrr")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--code_type", default='code', type=str,
                        help='use `code` or `code_tokens` in the json file to index.')

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as encoder_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=45,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--pred_model_dir", default=None, type=str,
                        help='model for prediction')
    parser.add_argument("--test_result_dir", default='test_results.tsv', type=str,
                        help='path to store test result')
    parser.add_argument("--test_predictions_output", default=None, type=str,
                        help="The output directory where the model predictions")
    parser.add_argument("--retrieval_predictions_output", default=None, type=str,
                        help="The output directory where the model predictions")
    parser.add_argument("--lang", default='', type=str, required=True,
                        help="Select dataset.")
    parser.add_argument("--predict_checkpoint_path", default=None, type=str,
                        help='model for calculating similarity')
    parser.add_argument("--t", default=1, type=float,
                        help='temperature for sharpening lambda')
    parser.add_argument("--weight_kl", default=1, type=float,
                        help='temperature for sharpening lambda')
    parser.add_argument("--weight_unif", default=1, type=float,
                        help='temperature for sharpening lambda')
    parser.add_argument("--bm25", action='store_true',
                        help="Use BM25 for weight estimation.")
    parser.add_argument("--self", action='store_true',
                        help="Use trained model for weight estimation.")
    parser.add_argument("--simcse", action='store_true',
                        help="Use unsupervised SimCSE model for weight estimation.")
    parser.add_argument("--softinfonce", action='store_true',
                        help="Use Soft-InfoNCE as loss function.")
    parser.add_argument("--bce", action='store_true',
                        help="Use Binary Cross Entropy as loss function.")
    parser.add_argument("--weightedinfonce", action='store_true',
                        help="Use weighted InfoNCE as loss function.")
    parser.add_argument("--klregularization", action='store_true',
                        help="Use InfoNCE + KL Div regularization as loss function.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # args.n_gpu = torch.cuda.device_count()
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.encoder_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.encoder_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.max_seq_length <= 0:
        args.max_seq_length = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.max_seq_length = min(args.max_seq_length, tokenizer.max_len_single_sentence)
    if args.encoder_name_or_path:
        model = model_class.from_pretrained(args.encoder_name_or_path,
                                            from_tf=bool('.ckpt' in args.encoder_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model_train = ModelContra(model, config, tokenizer, args)

    nl_tokenizer = None
    if args.self:
        model_predict = ModelContra(model, config, tokenizer, args)
        model_predict.load_state_dict(torch.load(os.path.join(args.predict_checkpoint_path, 'pytorch_model.bin')), strict=False)

    if args.simcse:
        nl_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-dot-v5")
        model_predict = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-dot-v5")
        model_predict.load_state_dict(torch.load('./model/simcse-cos0.07/checkpoint-best-mrr/pytorch_model.bin'))


    if args.checkpoint_path:
        logger.info('Reload from {} using args.checkpoint_path'.format(os.path.join(args.checkpoint_path, 'pytorch_model.bin')))
        if not args.no_cuda:
            model_train.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin')))
        else:
            model_train.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        train_data_dir = os.path.join(args.data_dir, args.lang)
        train_data_path = os.path.join(train_data_dir, args.train_data_file)
        train_dataset = TextDataset(tokenizer, args, train_data_path, nl_tokenizer)
        if args.simcse or args.self:
            train(args, train_dataset, model_train, tokenizer, model_predict)
        else:
            train(args, train_dataset, model_train, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-last'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        # output_dir = args.output_dir
        if torch.cuda.is_available():
            model_train.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        else:
            model_train.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        tokenizer = tokenizer.from_pretrained(output_dir)
        model_train.to(args.device)
        mrr = retrieval_CSN(args, model_train, tokenizer)
        logger.info("***** Eval results *****")
        logger.info("  Eval MRR = %s", str(mrr))
        logger.info("Eval Model From: {}".format(os.path.join(output_dir, 'pytorch_model.bin')))
        logger.info("***** Eval results *****")

    if args.do_retrieval and args.local_rank in [-1, 0]:

        logger.info("***** Retrieval results *****")
        checkpoint_prefix = 'checkpoint-best-mrr'
        if checkpoint_prefix not in args.output_dir and \
                os.path.exists(os.path.join(args.output_dir, checkpoint_prefix)):
            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        else:
            output_dir = args.output_dir
        if not args.pred_model_dir:
            model_path = os.path.join(output_dir, 'pytorch_model.bin')
        else:
            model_path = os.path.join(args.pred_model_dir, 'pytorch_model.bin')
        logger.info(model_path)
        model_train.load_state_dict(torch.load(model_path, map_location=args.device))
        tokenizer = tokenizer.from_pretrained(output_dir)
        model_train.to(args.device)
        mrr = retrieval_CSN(args, model_train, tokenizer)
        logger.info("Test Model From: {}".format(model_path))

    return results


if __name__ == "__main__":
    main()


