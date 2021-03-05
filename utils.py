import logging
import numpy as np
import torch
from torch import Tensor, distributed


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


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_len, tokenizer):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

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


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
    rt /= distributed.get_world_size()
    return rt


if __name__ == '__main__':
    # pickle_data()
    pass