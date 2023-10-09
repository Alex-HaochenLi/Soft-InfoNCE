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

logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm, trange
import multiprocessing
cpu_cont = 16


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 nl_tokens,
                 nl_ids,
    ):
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids


def convert_examples_to_features_retrieval_query(item):
    js,tokenizer,args=item
    #nl
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:args.nl_length-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length

    return InputFeatures(nl_tokens,nl_ids)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, root_path=None, pool=None):
        self.args = args
        self.examples = []
        data = []
        for lang in ['go', 'python', 'php', 'java', 'javascript', 'ruby']:
            file_path = os.path.join(os.path.join(root_path, lang), 'train.jsonl')
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    data.append((js, tokenizer, args))
        self.examples = pool.map(convert_examples_to_features_retrieval_query, tqdm(data, total=len(data)))

        for idx, example in enumerate(self.examples[:3]):
            logger.info("*** Example ***")
            logger.info("idx: {}".format(idx))
            logger.info("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
            logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item].nl_ids)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--root_path", default=None, type=str, required=True,
                        help="Root path.")
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    pool = multiprocessing.Pool(cpu_cont)

    # print arguments
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-dot-v5")
    model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-dot-v5")
    model.to(args.device)

    train_dataset = TextDataset(tokenizer, args, args.root_path, pool)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * args.num_train_epochs)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)

    model.zero_grad()
    model.train()

    tr_num, tr_loss, best_mrr = 0, 0, 0
    for idx in range(args.num_train_epochs):
        total_mrr = 0
        for step, nls in enumerate(train_dataloader):
            # get inputs
            nls = nls.to(args.device)

            def mean_pooling(model_output, attention_mask):
                token_embeddings = model_output.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                          min=1e-9)

            queries1 = model(nls, return_dict=True)
            queries1 = mean_pooling(queries1, nls.ne(0))
            queries2 = model(nls, return_dict=True)
            queries2 = mean_pooling(queries2, nls.ne(0))

            queries1 = torch.nn.functional.normalize(queries1, dim=1, p=2)
            queries2 = torch.nn.functional.normalize(queries2, dim=1, p=2)
            scores = torch.matmul(queries1, queries2.T)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores / 0.07, torch.arange(nls.size(0), device=scores.device))

            tr_loss += loss.item()
            tr_num += 1
            if (step + 1) % (len(train_dataloader) // 5) == 0:
                correct_scores = scores.diag().unsqueeze(1)
                ranks = (scores >= correct_scores).sum(1)
                mrr = torch.mean(1 / ranks).item()
                total_mrr += mrr

                logger.info("epoch {} step {} loss {} mrr {}".format(idx, step + 1, round(tr_loss / tr_num, 5), round(mrr, 5)))
                tr_loss = 0
                tr_num = 0

            # backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        checkpoint_prefix = 'checkpoint-last'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_dir = os.path.join(output_dir, '{}'.format('pytorch_model.bin'))
        torch.save(model_to_save.state_dict(), output_dir)

        # save best model
        if (total_mrr / 5) > best_mrr:
            best_mrr = (total_mrr / 5)
            logger.info("  " + "*" * 20)
            logger.info("  Best mrr:%s", round(best_mrr, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir = os.path.join(output_dir, '{}'.format('pytorch_model.bin'))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

    return 0


if __name__ == "__main__":
    main()