import logging
from logger import Logger
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
from torch import Tensor
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
from torch.optim.lr_scheduler import _LRScheduler, Optimizer
# from tqdm import tqdm_notebook as tqdm
import json
from apex import amp

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from sklearn.metrics import roc_curve, auc

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import distributed
from pytorch_pretrained_bert.optimization import BertAdam

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

# BERT_PRETRAINED_PATH = Path('chinese_L-12_H-768_A-12/')
# for file in os.listdir(BERT_PRETRAINED_PATH):
#     print(file)
model_path = Path('mode/')
model_path.mkdir(exist_ok=True)
# PYTORCH_PRETRAINED_BERT_CACHE = BERT_PRETRAINED_PATH / 'cache/'
# PYTORCH_PRETRAINED_BERT_CACHE.mkdir(exist_ok=True)

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
    "num_train_epochs": 5,
    "warmup_proportion": 0.1,
    # "local_rank": -1,
    "local_rank": 0,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    # "fp16": True,
    "loss_scale": 128,
}

# output_model_file = os.path.join(BERT_PRETRAINED_PATH, "pytorch_model.bin")
#
# # Load a trained model that you have fine-tuned
# model_state_dict = torch.load(output_model_file)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and val examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    '''
    input_ids：标记化文本的数字id列表
    input_mask：对于真实标记将设置为1，对于填充标记将设置为0
    segment_ids：分类问题，单句，均为0  下一句预测问题，第一句是0 第二句是1
    label_ids：文本的one-hot编码标签
    '''

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the val set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the val set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class KeySentenceProcessor(DataProcessor):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = [0, 1]

    def get_train_examples(self, data_dir, content, label, size=-1):
        # file = 'train'
        # data_dir = os.path.join(data_dir, file)
        if size == -1:
            return self._create_examples(content, label)

    def get_dev_examples(self, data_dir, content, label, size=-1):
        # file = 'dev'
        # data_dir = os.path.join(data_dir, file)
        if size == -1:
            return self._create_examples(content, label)

    def get_test_examples(self, data_dir, content, label, size=-1):
        # file = 'test'
        # data_dir = os.path.join(data_dir, file)
        if size == -1:
            return self._create_examples(content, label)

    def _create_examples(self, content, label):
        examples = []
        for index, item in enumerate(label):
            guid = index
            text_a = content[index]
            labels = item
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

    def get_labels(self):
        # with open("labor/tags.txt", 'r', encoding='utf-8') as fr:
        #     # 较为保险的操作，以防读取的标签含有空格
        #     labels = [x.strip() for x in fr.readlines()]
        return [0, 1]


def convert_examples_to_features(examples, label_list, max_seq_len, tokenizer):
    # label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        labels_ids = []
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        # 针对序列长度大于规定长度的情况，直接进行截断
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
        else:
            if len(tokens_a) > max_seq_len - 2:
                tokens_a = tokens_a[:max_seq_len - 2]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)


        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        # if example.labels == "0":
        #     labels_ids.append([1, 0])
        # else:
        #     labels_ids.append([0, 1])

        if example.labels == "0":
            labels_ids.append([0])
        else:
            labels_ids.append([1])

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=labels_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:

        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    # BertConfig
    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # self.device = device
        # self.rnn = nn.LSTM(input_size=config.emb_size,
        #                    hidden_size=config.encoder_hidden_size,
        #                    num_layers=config.num_layers,
        #                    dropout=config.hidden_dropout_prob,
        #                    bidirectional=config.bidirec)

        # self.rnn = nn.LSTM(input_size=config.hidden_size,
        #                    hidden_size=256,
        #                    num_layers=2,
        #                    dropout=config.hidden_dropout_prob,
        #                    bidirectional=True)

        # self.rnn = nn.GRU()
        self.classifier = torch.nn.Linear(768, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # 跑bert模型
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)

        # 使用rnn提取特征
        # output, _ = self.rnn(pooled_output)

        # output = self.dropout(output)
        # 对提取的特征进行全连接分类
        logits = self.classifier(pooled_output)

        # if labels is not None:
        #     loss_fct = BCEWithLogitsLoss().to(self.device)
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
        #     # loss = loss_fct(logits.view(-1, self.num_labels), labels)
        #     return loss
        # else:
        #     return logits
        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return sum(labels == outputs)


def fbeta(y_pred: Tensor, y_true: Tensor, thresh: float = 0.2, beta: float = 2, eps: float = 1e-9,
          sigmoid: bool = True):
    # Computes the f_beta between `preds` and `targets`
    beta2 = beta ** 2
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.float()
    TP = (y_pred * y_true).sum(dim=1)
    prec = TP / (y_pred.sum(dim=1) + eps)
    rec = TP / (y_true.sum(dim=1) + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean().item()


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def gen_data(Path):
    sentences, labels = [], []
    for (index, file) in enumerate(os.listdir(Path)):
        with open(os.path.join(Path, file), "r", encoding="utf-8") as fout:
            lines = fout.readlines()
            for line in lines:
                content = line.split("     ")
                if len(content) < 2:
                    continue
                sentences.append(content[1])
                labels.append(content[0])
    # print()
    return sentences, labels

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

        if args["local_rank"] != -1:
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

                if args["local_rank"] != -1:
                    tmp_train_loss = reduce_tensor(tmp_train_loss)
                    tmp_train_accuracy = reduce_tensor(torch.tensor(tmp_train_accuracy).to(device))

                if args['gradient_accumulation_steps'] > 1:
                    tmp_train_loss = tmp_train_loss / args['gradient_accumulation_steps']

                # if args['fp16']:
                #     optimizer.backward(tmp_train_loss)
                # else:
                #     tmp_train_loss.backward()

                optimizer.zero_grad()
                with amp.scale_loss(tmp_train_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

                # if (index + 1) % args['gradient_accumulation_steps'] == 0:

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

            # logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
            # logger.info('Accuracy after epoc {}'.format(tr_accuracy / nb_tr_examples))
            # logger.info('Eval after epoc {}'.format(i_ + 1))

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
                    if args["local_rank"] == -1:
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

        if args["fp16"]:
            #             scheduler.batch_step()
            # modify learning rate with special warm up BERT uses
            lr_this_step = args['learning_rate'] * warmup_linear(global_step / t_total, args['warmup_proportion'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
        else:
            scheduler.step()


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()
    return rt

# def predict(model, path, test_filename='test.csv'):
#     # labellist = ['过度医疗', '麻醉', '医疗器械', '其他和管理问题', '非手术治疗', '整形', '输液',
#     #              '护理', '输血', '资质', '用药', '检查检验', '生产', '文书',
#     #              '诊断', '失职', '医患沟通', '手术']
#     predict_processor = KeySentenceProcessor(path)
#     test_examples = predict_processor.get_test_examples(path, X_test, y_test, size=-1)
#
#     # Hold input data for returning it
#     input_data = [{'id': input_example.guid} for input_example in test_examples]
#
#     test_features = convert_examples_to_features(
#         test_examples, label_list, args['max_seq_length'], tokenizer)
#
#     logger.info("***** Running prediction *****")
#     logger.info("  Num examples = %d", len(test_examples))
#     logger.info("  Batch size = %d", args['eval_batch_size'])
#
#     all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
#     all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
#     all_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.float)
#     test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#
#     # Run prediction for full data
#     test_sampler = SequentialSampler(test_data)
#     test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])
#
#     all_logits = None
#
#     model.eval()
#     eval_loss, eval_accuracy = 0, 0
#     nb_eval_steps, nb_eval_examples = 0, 0
#     i = 0
#     for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
#         input_ids, input_mask, segment_ids, label_ids = batch
#         input_ids = input_ids.to(device)
#         input_mask = input_mask.to(device)
#         segment_ids = segment_ids.to(device)
#         label_ids = label_ids.to(device)
#
#         with torch.no_grad():
#             logits = model(input_ids, segment_ids, input_mask)
#
#         tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
#         logits = logits.sigmoid()
#         for i, logit in enumerate(logits):
#             for index, item in enumerate(logit):
#                 if item > 0.5:
#                     print(labellist[index])
#             print(',')
#             for index, item in enumerate(label_ids[i]):
#                 if item == 1.0:
#                     print(labellist[index])
#             print('\n')
#
#         if all_logits is None:
#             all_logits = logits.detach().cpu().numpy()
#         else:
#             all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
#         eval_accuracy += tmp_eval_accuracy
#         nb_eval_examples += input_ids.size(0)
#         nb_eval_steps += 1
#
#     eval_accuracy = eval_accuracy / nb_eval_examples
#
#     return all_logits, eval_accuracy


processors = {
    "textmultilabel": KeySentenceProcessor
}


if args["local_rank"] == -1 or args["no_cuda"]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    device_id = []
    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        device_id.append(i)
else:
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    distributed.init_process_group(backend='nccl')
    args["local_rank"] = torch.distributed.get_rank()
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
# PATH = '过失行为数据集'
# DataPath="labor/data_small_selected.json"
# LabelDict = dict()
content, labellist = gen_data(args["source_data_dir"])
print("共有数据:{}".format(len(content)))

X_train, X_test, y_train, y_test = train_test_split(content, labellist, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

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
                         weight_decay=args["weight_decay"])

# scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5, step_size=2500, last_batch_iteration=0)
# Eval Fn

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

model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
# loss_fct = torch.nn.functional.cross_entropy().to(device)
loss_fct = CrossEntropyLoss().to(device)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.2)

# class CyclicLR(object):
#     """Sets the learning rate of each parameter group according to
#     cyclical learning rate policy (CLR). The policy cycles the learning
#     rate between two boundaries with a constant frequency, as detailed in
#     the paper `Cyclical Learning Rates for Training Neural Networks`_.
#     The distance between the two boundaries can be scaled on a per-iteration
#     or per-cycle basis.
#     Cyclical learning rate policy changes the learning rate after every batch.
#     `batch_step` should be called after a batch has been used for training.
#     To resume training, save `last_batch_iteration` and use it to instantiate `CycleLR`.
#     This class has three built-in policies, as put forth in the paper:
#     "triangular":
#         A basic triangular cycle w/ no amplitude scaling.
#     "triangular2":
#         A basic triangular cycle that scales initial amplitude by half each cycle.
#     "exp_range":
#         A cycle that scales initial amplitude by gamma**(cycle iterations) at each
#         cycle iteration.
#     This implementation was adapted from the github repo: `bckenstler/CLR`_
#     Args:
#         optimizer (Optimizer): Wrapped optimizer.
#         base_lr (float or list): Initial learning rate which is the
#             lower boundary in the cycle for eachparam groups.
#             Default: 0.001
#         max_lr (float or list): Upper boundaries in the cycle for
#             each parameter group. Functionally,
#             it defines the cycle amplitude (max_lr - base_lr).
#             The lr at any cycle is the sum of base_lr
#             and some scaling of the amplitude; therefore
#             max_lr may not actually be reached depending on
#             scaling function. Default: 0.006
#         step_size (int): Number of training iterations per
#             half cycle. Authors suggest setting step_size
#             2-8 x training iterations in epoch. Default: 2000
#         mode (str): One of {triangular, triangular2, exp_range}.
#             Values correspond to policies detailed above.
#             If scale_fn is not None, this argument is ignored.
#             Default: 'triangular'
#         gamma (float): Constant in 'exp_range' scaling function:
#             gamma**(cycle iterations)
#             Default: 1.0
#         scale_fn (function): Custom scaling policy defined by a single
#             argument lambda function, where
#             0 <= scale_fn(x) <= 1 for all x >= 0.
#             mode paramater is ignored
#             Default: None
#         scale_mode (str): {'cycle', 'iterations'}.
#             Defines whether scale_fn is evaluated on
#             cycle number or cycle iterations (training
#             iterations since start of cycle).
#             Default: 'cycle'
#         last_batch_iteration (int): The index of the last batch. Default: -1
#     Example:
#         >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#         >>> scheduler = torch.optim.CyclicLR(optimizer)
#         >>> data_loader = torch.utils.data.DataLoader(...)
#         >>> for epoch in range(10):
#         >>>     for batch in data_loader:
#         >>>         scheduler.batch_step()
#         >>>         train_batch(...)
#     .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
#     .. _bckenstler/CLR: https://github.com/bckenstler/CLR
#     """
#
#     def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
#                  step_size=2000, mode='triangular', gamma=1.,
#                  scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):
#
#         #         if not isinstance(optimizer, Optimizer):
#         #             raise TypeError('{} is not an Optimizer'.format(
#         #                 type(optimizer).__name__))
#         self.optimizer = optimizer
#
#         if isinstance(base_lr, list) or isinstance(base_lr, tuple):
#             if len(base_lr) != len(optimizer.param_groups):
#                 raise ValueError("expected {} base_lr, got {}".format(
#                     len(optimizer.param_groups), len(base_lr)))
#             self.base_lrs = list(base_lr)
#         else:
#             self.base_lrs = [base_lr] * len(optimizer.param_groups)
#
#         if isinstance(max_lr, list) or isinstance(max_lr, tuple):
#             if len(max_lr) != len(optimizer.param_groups):
#                 raise ValueError("expected {} max_lr, got {}".format(
#                     len(optimizer.param_groups), len(max_lr)))
#             self.max_lrs = list(max_lr)
#         else:
#             self.max_lrs = [max_lr] * len(optimizer.param_groups)
#
#         self.step_size = step_size
#
#         if mode not in ['triangular', 'triangular2', 'exp_range'] \
#                 and scale_fn is None:
#             raise ValueError('mode is invalid and scale_fn is None')
#
#         self.mode = mode
#         self.gamma = gamma
#
#         if scale_fn is None:
#             if self.mode == 'triangular':
#                 self.scale_fn = self._triangular_scale_fn
#                 self.scale_mode = 'cycle'
#             elif self.mode == 'triangular2':
#                 self.scale_fn = self._triangular2_scale_fn
#                 self.scale_mode = 'cycle'
#             elif self.mode == 'exp_range':
#                 self.scale_fn = self._exp_range_scale_fn
#                 self.scale_mode = 'iterations'
#         else:
#             self.scale_fn = scale_fn
#             self.scale_mode = scale_mode
#
#         self.batch_step(last_batch_iteration + 1)
#         self.last_batch_iteration = last_batch_iteration
#
#     def batch_step(self, batch_iteration=None):
#         if batch_iteration is None:
#             batch_iteration = self.last_batch_iteration + 1
#         self.last_batch_iteration = batch_iteration
#         for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
#             param_group['lr'] = lr
#
#     def _triangular_scale_fn(self, x):
#         return 1.
#
#     def _triangular2_scale_fn(self, x):
#         return 1 / (2. ** (x - 1))
#
#     def _exp_range_scale_fn(self, x):
#         return self.gamma ** (x)
#
#     def get_lr(self):
#         step_size = float(self.step_size)
#         cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
#         x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)
#
#         lrs = []
#         param_lrs = zip(self.optimizer.param_groups, self.base_lrs, self.max_lrs)
#         for param_group, base_lr, max_lr in param_lrs:
#             base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
#             if self.scale_mode == 'cycle':
#                 lr = base_lr + base_height * self.scale_fn(cycle)
#             else:
#                 lr = base_lr + base_height * self.scale_fn(self.last_batch_iteration)
#             lrs.append(lr)
#         return lrs


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
# output_model_file.mkdir(exist_ok=True)
fit()


# torch.save(model_to_save.state_dict(), output_model_file)

# # Load a trained model that you have fine-tuned
# model_state_dict = torch.load(output_model_file)
# model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=num_labels,
#                                                                 state_dict=model_state_dict)
# model.to(device)
#
# # print(model)
# eval()
#
# result, eval_accuracy = predict(model, DATA_PATH)
# # 测试集上的准确率
# print(eval_accuracy)