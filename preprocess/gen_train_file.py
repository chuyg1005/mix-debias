from utils import *
import os, json
from argparse import ArgumentParser
from tqdm import tqdm
import random


def gen_co_data(filepath, mode):
    if not os.path.exists(filepath):
        print(f"{filepath} not exists.")
        assert 0

    data = json.load(open(filepath, "r"))

    co_data = []
    for item in tqdm(data):
        co_item = extract_context_only(item, mode=mode)
        co_data.append(co_item)

    with open(filepath.replace(".json", f"-{mode}.json"), "w") as f:
        json.dump(co_data, f)


def gen_eo_data(filepath, mode):
    if not os.path.exists(filepath):
        print(f"{filepath} not exists.")
        assert 0

    data = json.load(open(filepath, "r"))

    eo_data = []
    for item in tqdm(data):
        eo_item = extract_entity_only(item, mode=mode)
        eo_data.append(eo_item)

    with open(filepath.replace(".json", f"-{mode}.json"), "w") as f:
        json.dump(eo_data, f)


def gen_aug_dataset(filepath, k):
    if not os.path.exists(filepath):
        print(f"{filepath} not exists.")
        assert 0

    data = json.load(open(filepath, "r"))

    entity_dict = gen_entity_dict(data)

    print("generating augmented-dataset by entity-switch...")
    aug_data = []
    for item in tqdm(data):
        # [item, entity_only_item, context_only_item, new_item1, new_item2, ...]
        aug_item = [item]
        entity_only_item = extract_entity_only(item, mode='eo')
        context_only_item = extract_context_only(item, mode='co')
        aug_item += [entity_only_item, context_only_item]
        subj_type = item["subj_type"]
        obj_type = item["obj_type"]
        for _ in range(k):
            new_subj = random.choice(entity_dict[subj_type])
            new_obj = random.choice(entity_dict[obj_type])
            new_item = substitute_item_with_new_entities(item, new_subj, new_obj)
            aug_item.append(new_item)
        aug_data.append(aug_item)

    filedir = os.path.dirname(filepath)
    with open(os.path.join(filedir, f"train4debias.json"), "w") as f:
        json.dump(aug_data, f)

def main(args):
    random.seed(args.seed)

    if args.mode.startswith('co'):
        gen_co_data(os.path.join(args.data_root, args.dataset, f"{args.split}.json"), args.mode)
    elif args.mode.startswith('eo'):
        gen_eo_data(os.path.join(args.data_root, args.dataset, f"{args.split}.json"), args.mode)
    elif args.mode == 'aug':
        assert args.split == 'train', "augmentation only works for train set"
        gen_aug_dataset(os.path.join(args.data_root, args.dataset, f"{args.split}.json"), args.k)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--mode", default='aug', type=str, choices=['co', 'eo', 'aug', 'eo-t', 'eo-m', 'co-o'])
    parser.add_argument("--split", default="train",
                        type=str)
    args = parser.parse_args()
    main(args)
