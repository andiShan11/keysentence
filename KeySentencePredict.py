import torch
from KeySentenceClassification import KeySentenceProcessor, convert_examples_to_features,

from pytorch_pretrained_bert.optimization import BertAdam
import os
from pathlib import Path
import argparse
from apex import amp

# BERT_PRETRAINED_PATH = Path('chinese_L-12_H-768_A-12/')
#
# PYTORCH_PRETRAINED_BERT_CACHE = BERT_PRETRAINED_PATH / 'cache/'
# PYTORCH_PRETRAINED_BERT_CACHE.mkdir(exist_ok=True)
# args = {
#     "bert_model": BERT_PRETRAINED_PATH,
#     "no_cuda": False
# }


def predict(model, path):
    predict_processor = KeySentenceProcessor(path)
    test_examples = predict_processor.get_test_examples(path, X_test, y_test, size=-1)

    # Hold input data for returning it
    # input_data = [{'id': input_example.guid} for input_example in test_examples]

    test_features = convert_examples_to_features(
        test_examples, label_list, args['max_seq_length'], tokenizer)

    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in test_features], dtype=torch.float)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])

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

        # # tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        # tmp_eval_accuracy = accuracy(logits, label_ids)
        # logits = logits.sigmoid()

        tmp_pre_loss = loss_fct(logits.view(-1, num_labels), label_ids.squeeze())
        tmp_pre_accuracy = accuracy(logits.view(-1, num_labels).detach().cpu().numpy(),
                                     label_ids.squeeze().detach().cpu().numpy())

        if args["local_rank"] != -1:
            tmp_pre_loss = reduce_tensor(tmp_pre_loss)
            tmp_pre_accuracy = reduce_tensor(torch.tensor(tmp_pre_accuracy).to(device))

        pre_loss += tmp_pre_loss.mean().item()
        pre_accuracy += tmp_pre_accuracy.item()
        nb_pre_examples += input_ids.size(0)
        nb_pre_steps += 1

    pre_loss = pre_loss / nb_pre_steps
    pre_accuracy = pre_accuracy / nb_pre_examples

    result = {'pre_loss': pre_loss, 'pre_accuracy': pre_accuracy}

    output_pre_file = os.path.join(args['output_dir'], "pre_results.txt")
    with open(output_pre_file, "w") as writer:
        logger.info("***** Pre results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return pre_loss, pre_accuracy


parser = argparse.ArgumentParser(description="KeySentencePredict.py")
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed')
parser.add_argument('--no_cuda', default=False, type=bool, help='whether there are gpus')
parser.add_argument('--output_dir', default=CLAS_DATA_PATH / 'output')
args = parser.parse_args()

num_labels = 2
model_path = Path('mode/')
device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")

logger.info("local_rank:{}", args["local_rank"])

if args["local_rank"] == -1:
    output_model_file = os.path.join(model_path, "finetuned_pytorch_model.bin")
    model_state_dict = torch.load(output_model_file)
    model = BertForMultiLabelSequenceClassification.from_pretrained("bert-base-chinese",
                                                                    num_labels=num_labels,
                                                                    state_dict=model_state_dict)
else:
    model = BertForMultiLabelSequenceClassification.from_pretrained("ber-base-chinese",
                                                                    num_labels=num_labels)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args['learning_rate'],
                         warmup=args['warmup_proportion'],
                         # t_total=t_total,
                         weight_decay=args["weight_decay"])

    output_model_file = os.path.join(model_path, "amp_checkpoint.pt")
    checkpoint = torch.load(output_model_file)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])


model.to(device)


print("开始预测")
result, eval_accuracy = predict(model, DATA_PATH)
# 测试集上的准确率
print(eval_accuracy)
