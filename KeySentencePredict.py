from utils import KeySentenceProcessor, convert_examples_to_features, accuracy, reduce_tensor
from models import BertForMultiLabelSequenceClassification

from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
from torch import distributed
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
import argparse
import pickle
import logging
import os
from pathlib import Path


def predict(model, path):
    with open(os.path.join(Path('data'), "dev.pkl"), "rb") as fin:
        x_dev, y_dev = pickle.load(fin)
    dev_examples = predict_processor.get_test_examples(path, x_dev, x_dev, size=-1)
    # print("测试数据量：{}".format(len(dev_examples)))
    # print("device：{}".format(device))
    test_features = convert_examples_to_features(
        dev_examples, label_list, args.max_seq_length, tokenizer)

    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(dev_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pre_loss, pre_accuracy = 0, 0
    nb_pre_steps, nb_pre_examples = 0, 0
    for step, batch in enumerate(tqdm(test_dataloader, desc="Prediction Iteration")):
        input_ids, input_mask, segment_ids, label_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
        loss_fct = CrossEntropyLoss().to(device)
        tmp_pre_loss = loss_fct(logits.view(-1, num_labels), label_ids.squeeze())

        tmp_pre_accuracy = accuracy(logits.view(-1, num_labels).detach().cpu().numpy(),
                                     label_ids.squeeze().detach().cpu().numpy())

        if args.local_rank != -1:
            tmp_pre_loss = reduce_tensor(tmp_pre_loss)
            tmp_pre_accuracy = reduce_tensor(torch.tensor(tmp_pre_accuracy).to(device))

        pre_loss += tmp_pre_loss.mean().item()
        pre_accuracy += tmp_pre_accuracy.item()
        nb_pre_examples += input_ids.size(0)
        nb_pre_steps += 1

    pre_loss = pre_loss / nb_pre_steps
    pre_accuracy = pre_accuracy / nb_pre_examples

    result = {'pre_loss': pre_loss, 'pre_accuracy': pre_accuracy}

    output_pre_file = os.path.join(args.output_dir, "pre_results.txt")
    with open(output_pre_file, "w") as writer:
        logger.info("***** Pre results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return pre_loss, pre_accuracy


CLAS_DATA_PATH = Path('data/')
CLAS_DATA_PATH.mkdir(exist_ok=True)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="KeySentencePredict.py")
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed')
parser.add_argument('--no_cuda', default=False, type=bool, help='whether there are gpus')
parser.add_argument('--output_dir', default=CLAS_DATA_PATH / 'output')
parser.add_argument("--task_name", default="textmultilabel")
parser.add_argument("--weight_decay", default=1e-4)
parser.add_argument("--learning_rate", default=2e-5)
parser.add_argument("--warmup_proportion", default=0.1)
parser.add_argument("--max_seq_length", default=256)
parser.add_argument("--eval_batch_size", default=8)
args = parser.parse_args()


num_labels = 2
model_path = Path('mode/')

if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    print("local_rank:{}".format(args.local_rank))

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
predict_processor = KeySentenceProcessor("")
label_list = predict_processor.get_labels()
num_labels = len(label_list)

if args.local_rank == -1:
    output_model_file = os.path.join(model_path, "finetuned_pytorch_model.bin")
    model_state_dict = torch.load(output_model_file)
    model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-chinese",
                                                                    num_labels=num_labels,
                                                                    state_dict=model_state_dict['model'])
else:
    from apex.parallel import DistributedDataParallel as DDP

    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # #
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=args.learning_rate,
    #                      warmup=args.warmup_proportion,
    #                      t_total=-1,
    #                      weight_decay=args.weight_decay)
    # loc = 'cuda:{}'.format()
    output_model_file = os.path.join(model_path, "amp_checkpoint.pt")
    # loc = 'cuda:{}'.format(args.local_rank)
    checkpoint = torch.load(output_model_file)
    # model, _ = amp.initialize(model, opt_level='O0')
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # amp.load_state_dict(checkpoint['amp'])
    # output_model_file = os.path.join(model_path, "amp_checkpoint.pt")
    model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-chinese",
                                                                    num_labels=num_labels,
                                                                    state_dict=checkpoint['model'])
    model.to(device)
    model = DDP(model)

result, eval_accuracy = predict(model, "")
# 测试集上的准确率
print(eval_accuracy)
