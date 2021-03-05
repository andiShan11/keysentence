# keysentence
使用二分类方法，决定每个句子是否是关键句子
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 KeySentencePredict.py
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 12345 KeySentenceClassification.py
