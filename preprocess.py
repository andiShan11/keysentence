import os
import pickle
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def gen_data(Path):
    sentences, labels = [], []
    # ng_num = 0
    for (index, file) in enumerate(os.listdir(Path)):
        with open(os.path.join(Path, file), "r", encoding="utf-8") as fout:
            lines = fout.readlines()
            for line in lines:
                content = line.split("     ")
                if len(content) < 2:
                    continue
                # if content[0] != "1":
                #     ng_num = ng_num + 1
                # else:
                #     print(content[1], content[0])
                sentences.append(content[1])
                labels.append(content[0])
    return sentences, labels, ng_num


def pickle_data(args):
    sentences, labels, num = gen_data(args.source_data_dir)
    # print("共有标签0:{}".format(num))
    X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    with open(os.path.join(args.output_data_dir, 'train.pkl'), 'wb') as fout:
        pickle.dump((X_train, y_train), fout)

    with open(os.path.join(args.output_data_dir, 'test.pkl'), 'wb') as fout:
        pickle.dump((X_test, y_test), fout)

    with open(os.path.join(args.output_data_dir, 'dev.pkl'), 'wb') as fout:
        pickle.dump((X_dev, y_dev), fout)


def main():
    parser = argparse.ArgumentParser(description="utils.py")
    parser.add_argument('--source_data_dir', default="/home/yf/Documents/zs/关键句子_训练集")
    parser.add_argument('--output_data_dir', default=Path('data'))
    args = parser.parse_args()

    pickle_data(args=args)


if __name__ == '__main__':
    main()