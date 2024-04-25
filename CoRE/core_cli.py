import argparse
import os
import json
import numpy as np
from tqdm import tqdm


def get_micro_f1(keys, predictions, na=0):
    correct_by_relation = ((keys == predictions) & (predictions != na)).astype(np.int32).sum()
    guessed_by_relation = (predictions != na).astype(np.int32).sum()
    gold_by_relation = (keys != na).astype(np.int32).sum()

    prec_micro = 1.0
    if guessed_by_relation > 0:
        prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = 1.0
    if gold_by_relation > 0:
        recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    return prec_micro, recall_micro, f1_micro


def get_macro_f1():
    pass


def train(origin_predict, entity_predict, model_label2id):
    origin_labels = origin_predict[:, -1].astype(int)
    entity_labels = entity_predict[:, -1].astype(int)
    assert np.all(origin_labels == entity_labels)
    origin_probs = origin_predict[:, :-1]
    entity_probs = entity_predict[:, :-1]
    # 重新映射
    na = model_label2id['no_relation']
    # 生成label_mask
    label_mask = np.zeros_like(origin_probs)
    label_mask[:, na] = 1.


    # 网格搜索查找最合适的lamb1和lamb2
    max_f1 = 0.
    best_lamb1 = 0.
    best_lamb2 = 0.
    for lamb1 in tqdm(np.arange(-2, 2, 0.1)):
        for lamb2 in np.arange(-2, 2, 0.1):
            probs = origin_probs + lamb1 * entity_probs + lamb2 * label_mask
            predictions = np.argmax(probs, axis=-1)
            prec, recall, f1 = get_micro_f1(origin_labels, predictions, )
            # print(f"lamb1: {lamb1}, lamb2: {lamb2}, F1: {f1}")
            if f1 > max_f1:
                max_f1 = f1
                best_lamb1 = lamb1
                best_lamb2 = lamb2
    print(f"Best lamb1: {best_lamb1}, Best lamb2: {best_lamb2}")
    return best_lamb1, best_lamb2


def test(origin_predict, entity_predict, model_label2id, lamb1, lamb2):
    origin_labels = origin_predict[:, -1].astype(int)
    entity_labels = entity_predict[:, -1].astype(int)
    assert np.all(origin_labels == entity_labels)
    origin_probs = origin_predict[:, :-1]
    entity_probs = entity_predict[:, :-1]
    # 重新映射
    # label2id = {v: k for k, v in model_id2label.items()}
    na = model_label2id['no_relation']
    # 生成label_mask
    label_mask = np.zeros_like(origin_probs)
    label_mask[:, na] = 1.

    probs = origin_probs + lamb1 * entity_probs + lamb2 * label_mask
    predictions = np.argmax(probs, axis=-1)
    prec, recall, f1 = get_micro_f1(origin_labels, predictions, na=na)
    return prec, recall, f1


def main(args):
    model_pred_dir = os.path.join(args.pred_root, args.model_name, args.dataset, f"{args.mode}-{args.seed}")
    if not os.path.exists(model_pred_dir):
        assert False, f"Model prediction directory {model_pred_dir} not exists."
    label2id_path = os.path.join(model_pred_dir, 'label2id.json')
    if not os.path.exists(label2id_path):
        assert False, f"Label2id file {label2id_path} not exists."
    label2id = json.load(open(label2id_path, 'r'))

    dev_pred_path = os.path.join(model_pred_dir, f'{args.dev_name}.txt')
    dev_eo_pred_path = os.path.join(model_pred_dir, f'{args.dev_name}-eo.txt')
    test_pred_path = os.path.join(model_pred_dir, f'{args.test_name}.txt')
    test_eo_pred_path = os.path.join(model_pred_dir, f'{args.test_name}-eo.txt')
    if not os.path.exists(dev_pred_path):
        assert False, f"Dev prediction file {dev_pred_path} not exists."
    if not os.path.exists(dev_eo_pred_path):
        assert False, f"Dev EO prediction file {dev_eo_pred_path} not exists."
    if not os.path.exists(test_pred_path):
        assert False, f"Test prediction file {test_pred_path} not exists."
    if not os.path.exists(test_eo_pred_path):
        assert False, f"Test EO prediction file {test_eo_pred_path} not exists."

    dev_pred = np.loadtxt(dev_pred_path, dtype=float, delimiter=',')
    dev_eo_pred = np.loadtxt(dev_eo_pred_path, dtype=float, delimiter=',')
    test_pred = np.loadtxt(test_pred_path, dtype=float, delimiter=',')
    test_eo_pred = np.loadtxt(test_eo_pred_path, dtype=float, delimiter=',')

    print("=========================================")
    print(f"Model: {args.model_name}, Mode: {args.mode}, Seed: {args.seed}, Dataset: {args.dataset}")
    print(f"Dev Name: {args.dev_name}, Test Name: {args.test_name}")
    prec, recall, f1 = test(dev_pred, dev_eo_pred, label2id, 0., 0.)
    print(f"Before CoRE: Prec: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    lamb1, lamb2 = train(dev_pred, dev_eo_pred, label2id)
    prec, recall, f1 = test(test_pred, test_eo_pred, label2id, lamb1, lamb2)
    print(f"After CoRE: Prec: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", default="./data/predictions")
    parser.add_argument("--model_name", default="ibre", type=str, choices=["ibre", "luke"])
    parser.add_argument("--mode", default="default", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default='tacred', type=str, choices=['tacred', 'retacred'])
    parser.add_argument("--dev_name", default='dev', type=str)
    parser.add_argument("--test_name", default='test', type=str)

    args = parser.parse_args()
    main(args)
