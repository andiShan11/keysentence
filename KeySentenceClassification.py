import logging
from logger import Logger
from utils import convert_examples_to_features, KeySentenceProcessor, accuracy, warmup_linear, reduce_tensor
from models import BertForMultiLabelSequenceClassification
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, \
    BertForMaskedLM, BertForSequenceClassification
from pathlib import Path
import torch
from torch import optim
import re
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack

# from torch.nn import BCEWithLogitsLoss

# from fastai.text import Tokenizer, Vocab
# import pandas as pd

import collections
import os
import pdb
from tqdm import tqdm, trange
import sys
import random
import numpy as np
# import apex
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import _LRScheduler
# from tqdm import tqdm_notebook as tqdm
import json
from apex import amp
import pickle

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import distributed
from pytorch_pretrained_bert.optimization import BertAdam
import argparse

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

loggers = Logger('./logs')

DATA_PATH = Path('data/')
DATA_PATH.mkdir(exist_ok=True)

PATH = Path('data/tmp')
PATH.mkdir(exist_ok=True)

CLAS_DATA_PATH = PATH / 'class'
CLAS_DATA_PATH.mkdir(exist_ok=True)

model_state_dict = None

model_path = Path('mode/')
model_path.mkdir(exist_ok=True)


# parser = argparse.ArgumentParser(description="classification")
# parser.add_argument("--train_size", default=-1, type=int)
# parser.add_argument("--val_size", default=-1, type=int)
# parser.add_argument("--source_data_dir", default="/home/yf/Documents/zs/关键句子_训练集")
# parser.add_argument("--full_data_dir", default=DATA_PATH)
# parser.add_argument("--data_dir", default=PATH)
# parser.add_argument("--task_name", default="textmultilabel")
# parser.add_argument("--no_cuda", default=False)
# parser.add_argument("--output_dir", default=CLAS_DATA_PATH / 'output')
# parser.add_argument("--max_seq_length", default=256)
# parser.add_argument("--do_train", default=True)
# parser.add_argument("--do_eval", default=True)
# parser.add_argument("--do_lower_case", default=True)
# parser.add_argument("--train_batch_size", default=8)
# parser.add_argument("--eval_batch_size", default=8)
# parser.add_argument("--learning_rate", default=2e-5)
# parser.add_argument("--weight_decay", default=1e-4)
# parser.add_argument("--num_train_epochs", default=5)
# parser.add_argument("warmup_proportion", default=0.1)
# parser.add_argument("local_rank", default=0)
# parser.add_argument("seed", default=42)
# parser.add_argument("gradient_accumulation_steps", default=1)
# parser.add_argument("optimize_on_cpu", default=False)
# parser.add_argument("fp16", default=False)
# parser.add_argument("loss_scale", default=128)
# args = parser.parse_args()

args = {
    "train_size": -1,
    "val_size": -1,
    "source_data_dir": "/home/yf/Documents/zs/关键句子_训练集",
    "full_data_dir": DATA_PATH,
    "data_dir": PATH,
    "task_name": "textmultilabel",
    "no_cuda": False,
    # "bert_model": BERT_PRETRAINED_PATH,
    "output_dir": CLAS_DATA_PATH / 'output',
    "max_seq_length": 256,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "learning_rate": 2e-5,
    "weight_decay": 1e-4,
    "num_train_epochs": 1,
    "warmup_proportion": 0.1,
    # "local_rank": -1,
    "local_rank": 0,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128,
}


# Prepare model
def get_model():
    #     pdb.set_trace()
    if model_state_dict:
        model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_labels,
                                                                        state_dict=model_state_dict)
    else:
        model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_labels,
                                                                        cache_dir=None)
    return model


def eval():
    args['output_dir'].mkdir(exist_ok=True)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args['max_seq_length'], tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            # tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.squeeze())
        tmp_eval_accuracy = accuracy(logits.view(-1, num_labels).detach().cpu().numpy(),
                                     label_ids.squeeze().detach().cpu().numpy())

        if args['local_rank'] != -1:
            tmp_eval_loss = reduce_tensor(tmp_eval_loss)
            tmp_eval_accuracy = reduce_tensor(torch.tensor(tmp_eval_accuracy).to(device))

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy.item()
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss, 'eval_accuracy': eval_accuracy}

    output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return eval_loss, eval_accuracy


def fit(num_epoch=args['num_train_epochs']):
    global_step = 0
    model.train()
    for i_ in tqdm(range(int(num_epoch)), desc="Epoch"):
        print('当前阶段******************************', i_)
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for index, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            try:
                logits = model(input_ids, segment_ids, input_mask, label_ids)

                tmp_train_loss = loss_fct(logits.view(-1, num_labels), label_ids.squeeze())
                tmp_train_accuracy = accuracy(logits.view(-1, num_labels).detach().cpu().numpy(),
                                              label_ids.squeeze().detach().cpu().numpy())
                if n_gpu > 1:
                    tmp_train_loss = tmp_train_loss.mean()  # mean() to average on multi-gpu.

                if args['local_rank'] != -1:
                    tmp_train_loss = reduce_tensor(tmp_train_loss)
                    tmp_train_accuracy = reduce_tensor(torch.tensor(tmp_train_accuracy).to(device))

                if args['gradient_accumulation_steps'] > 1:
                    tmp_train_loss = tmp_train_loss / args['gradient_accumulation_steps']

                if args["local_rank"] != -1:
                    with amp.scale_loss(tmp_train_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    tmp_train_loss.backward()

                optimizer.zero_grad()
                optimizer.step()

                tr_loss += tmp_train_loss.item()
                tr_accuracy += tmp_train_accuracy.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                global_step += 1
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e

            # Tensorboard Logging
            eval_loss, eval_accuracy = 0, 0
            if global_step % 500 == 0:
                eval_loss, eval_accuracy = eval()

                logger.info('tr_loss:{} & tr_accuracy:{}'.format(tr_loss / nb_tr_steps, tr_accuracy / nb_tr_examples))
                logger.info('eval_loss:{} & eval_accuracy:{}'.format(eval_loss, eval_accuracy))
                info = {'tr_loss': tr_loss / nb_tr_steps, 'tr_accuracy': tr_accuracy / nb_tr_examples}
                for tag, value in info.items():
                    loggers.scalar_summary(tag, value, global_step + 1)
                info = {'eval_loss': eval_loss, 'eval_accuracy': eval_accuracy}
                for tag, value in info.items():
                    loggers.scalar_summary(tag, value, global_step + 1)

            # 将模型保存下来
            if global_step % 1000 == 0:
                params.append(eval_accuracy)
                if eval_accuracy >= max(params):
                    if args['local_rank'] == -1:
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(model_path, "finetuned_pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                    else:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'amp': amp.state_dict()
                        }
                        output_model_file = os.path.join(model_path, "amp_checkpoint.pt")
                        torch.save(checkpoint, output_model_file)

        if args['fp16']:
            #             scheduler.batch_step()
            # modify learning rate with special warm up BERT uses
            lr_this_step = args['learning_rate'] * warmup_linear(global_step / t_total, args['warmup_proportion'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
        else:
            scheduler.step()


processors = {
    "textmultilabel": KeySentenceProcessor
}

logger.info("local_rank是-----------------------------------:{}".format(args['local_rank']))

if args['local_rank'] == -1 or args['no_cuda']:
    device = torch.device("cuda" if torch.cuda.is_available() and not args['no_cuda'] else "cpu")
    device_id = []
    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        device_id.append(i)
else:
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    distributed.init_process_group(backend='nccl')
    args['local_rank'] = torch.distributed.get_rank()
    torch.cuda.set_device(args['local_rank'])
    device = torch.device("cuda", args['local_rank'])
    n_gpu = 1


logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    device, n_gpu, bool(args['local_rank'] != -1), args['fp16']))

args['train_batch_size'] = int(args['train_batch_size'] / args['gradient_accumulation_steps'])

# 设置随机种子
random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
if n_gpu > 0:
    torch.cuda.manual_seed_all(args['seed'])

task_name = args['task_name'].lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name](args['data_dir'])
label_list = processor.get_labels()
num_labels = len(label_list)

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
train_examples = None
num_train_steps = None


with open(os.path.join(Path('data'), "train.pkl"), "rb") as fin:
    X_train, y_train = pickle.load(fin)


if args['do_train']:
    train_examples = processor.get_train_examples(args['full_data_dir'], X_train, y_train, size=args['train_size'])
    #     train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
    num_train_steps = int(
        len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])


model = get_model()

if args['fp16']:
    model.half()
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
t_total = num_train_steps
if args['local_rank'] != -1:
    t_total = t_total // distributed.get_world_size()

if args['fp16']:
    try:
        from apex.optimizers import FP16_Optimizer
        from apex.optimizers import FusedAdam
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

    optimizer = FusedAdam(optimizer_grouped_parameters,
                          lr=args['learning_rate'],
                          bias_correction=False)
    if args['loss_scale'] == 0:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    else:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args['loss_scale'])

else:
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args['learning_rate'],
                         warmup=args['warmup_proportion'],
                         t_total=t_total,
                         weight_decay=args['weight_decay'])

# scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5, step_size=2500, last_batch_iteration=0)
# Eval Fn
loss_fct = CrossEntropyLoss()
if args['local_rank'] != -1:
    # try:
    from apex.parallel import DistributedDataParallel as DDP
    # except ImportError:
    #     raise ImportError(
    #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    # from torch.nn.parallel import DistributedDataParallel as DDP
    # model = DDP(model, device_ids=[args["local_rank"]])
    model = DDP(model, delay_allreduce=True)
elif n_gpu > 1:
    print('n_gpu:', n_gpu)
    model = torch.nn.DataParallel(model, device_ids=device_id, output_device=device_id[0])
if args['local_rank'] != -1:
    model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
# loss_fct = torch.nn.functional.cross_entropy().to(device)
    loss_fct = loss_fct.to(device)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.2)


with open(os.path.join(Path('data'), "dev.pkl"), "rb") as fin:
    X_dev, y_dev = pickle.load(fin)

eval_examples = processor.get_dev_examples(args['full_data_dir'], X_dev, y_dev, size=args['val_size'])
train_features = convert_examples_to_features(
    train_examples, label_list, args['max_seq_length'], tokenizer)

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", args['train_batch_size'])
logger.info("  Num steps = %d", num_train_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)

train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

if args['local_rank'] == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])
# 不固定参数的值
model.module.unfreeze_bert_encoder()
# model.unfreeze_bert_encoder()
params = []
fit()
