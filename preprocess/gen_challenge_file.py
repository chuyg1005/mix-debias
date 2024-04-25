import numpy as np
import json
import random
from utils import extract_entity_only, substitute_item_with_new_entities, gen_entity_dict
from argparse import ArgumentParser
from evaluate import load

import os
import sys
import math
from tqdm import tqdm

tqdm.disable = True

print("Current working directory: ", os.getcwd())
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "RE_improved_baseline"))
from RE_improved_baseline.utils import predict, loadModel, set_seed
from RE_improved_baseline.prepro import DatasetProcessor


class Generator:
    def __init__(self, save_path, entity_dict, dataset, device='cuda', cand_size_v2=128, topk_v2=10, mode="challenge"):
        self.lm_name = 'gpt2'
        self.device = device
        self.mode = mode
        self.entity_dict = entity_dict
        # 使用gpt2作为语言模型计算句子的perplexity
        if self.mode == 'challenge':
            self.perplexity = load("perplexity", module_type="metric")

            # 使用mention-only的模型计算概率
            self.eo_model, tokenizer, _args = loadModel(save_path, device)
            self.processor = DatasetProcessor(_args, tokenizer)
            self.cand_size_v2 = cand_size_v2
            self.topk_v2 = topk_v2
        elif self.mode == 'ibre':  # 对模型进行counterfactual analysis，找出无法正确预测的hard样本
            self.model, tokenizer, _args = loadModel(save_path, device)
            self.processor = DatasetProcessor(_args, tokenizer)

    def generate_batch(self, items):
        if self.mode == 'shuffle':
            return self._generate_v1(items)
        elif self.mode == 'challenge':
            return self._generate_v2(items)
        elif self.mode == 'ibre':  # deprecated
            return self._generate_v0(items)

    def _generate_v0(self, items):
        """counterfactual analysis for the model, find the hard samples that the model can't predict correctly under the entity-only setting"""
        new_items = []
        items = [extract_entity_only(item) for item in items]
        features = self.processor.encode(items, show_bar=False)
        keys, preds = predict(self.model, features, len(items), self.device, False)
        for i in range(len(items)):
            if keys[i] != preds[i]:
                new_items.append(items[i])
        return new_items

    def _generate_v1(self, items):
        """generate new items by substituting entities in the original items with new entities from the entity_dict with entity-type constraints"""
        new_items = []
        for item in items:
            subj_type = item['subj_type']
            obj_type = item['obj_type']
            new_subj = random.choice(self.entity_dict[subj_type])
            new_obj = random.choice(self.entity_dict[obj_type])
            new_item = substitute_item_with_new_entities(item, new_subj, new_obj)
            new_items.append(new_item)
        return new_items

    def _generate_v2(self, items):
        n_items = len(items)
        batch_items = []
        for item in items:
            subj_type = item['subj_type']
            obj_type = item['obj_type']
            for i in range(self.cand_size_v2):
                new_subj = random.choice(self.entity_dict[subj_type])
                new_obj = random.choice(self.entity_dict[obj_type])
                new_item = substitute_item_with_new_entities(item, new_subj, new_obj)
                batch_items.append(new_item)

        batch_biases = self.computeBias(batch_items)
        topk = self.topk_v2
        # topk = self.cand_size_v2 // 10

        batch_items_new = []
        for i in range(0, n_items * self.cand_size_v2, self.cand_size_v2):
            items = batch_items[i:i + self.cand_size_v2]
            biases = batch_biases[i:i + self.cand_size_v2]
            indices = np.argsort(biases)[:topk]
            batch_items_new.extend([items[i] for i in indices])

        perplexities = self.computePerplexityByGPT2(batch_items_new)
        new_items = []
        for i in range(0, n_items * topk, topk):
            items = batch_items_new[i:i + topk]
            perplexities_ = perplexities[i:i + topk]
            index = np.argmin(perplexities_)
            new_items.append(items[index])

        return new_items

    def computeBias(self, items):
        """compute bias using the entity-only model"""
        new_items = []
        for item in items:
            new_item = extract_entity_only(item)
            new_items.append(new_item)

        features = self.processor.encode(new_items, show_bar=False)
        _, _, probs = predict(self.eo_model, features, len(new_items), self.device, True)
        labels = [feature['labels'] for feature in features]
        label_probs = []
        for i in range(len(labels)):
            label_probs.append(probs[i][labels[i]])

        return label_probs

    def computePerplexityByGPT2(self, items):
        sentences = [' '.join(item['token']) for item in items]
        perplexities = self.perplexity.compute(model_id=self.lm_name, add_start_token=True, predictions=sentences)
        return perplexities['perplexities']


def generateEntityDict(data_root, dataset, mode, split):
    if mode == 'challenge':
        print("Challenge mode, use entity-dict-wiki.json to generate entity dict")
        entity_dict_wiki = json.load(
            open(os.path.join(data_root, "entity-dict-wiki.json")))
    else:
        entity_dict_wiki = {}
    data_path = os.path.join(data_root, dataset, f"{split}.json")
    data = json.load(open(data_path))
    entity_dict = gen_entity_dict(data)
    # 合并来自wiki的实体
    for entity_type, entities in entity_dict_wiki.items():
        if entity_type in entity_dict:
            entity_dict[entity_type] = set(entity_dict[entity_type] + entities)

    # 将set转换为list，使得可以通过random.choice()随机选择
    new_entity_dict = {}
    for k, v in entity_dict.items():
        new_entity_dict[k] = list(v)

    # json.dump(new_entity_dict, open(entity_dict_save_path, 'w'), ensure_ascii=False, indent=2)

    return new_entity_dict


def main(args):
    entity_dict = generateEntityDict(args.data_root, args.dataset, args.mode, args.split)
    test_data_path = os.path.join(args.data_root, args.dataset, f"{args.split}.json")
    test_challenge_data_path = os.path.join(args.data_root, args.dataset,
                                            f"{args.split}_{args.mode}.json")
    if args.mode == 'challenge':
        save_path = os.path.join(args.ckpt_dir, args.dataset, f"EntityOnly-{args.seed}")
        print(f"Challenge mode, model_path: {save_path}")
    elif args.mode == 'shuffle':
        save_path = None
    elif args.mode == 'ibre':
        save_path = os.path.join(args.ckpt_dir, args.dataset, f"default-{args.seed}")
        print(f"IBRE mode, model_path: {save_path}")
    else:
        assert 0, f"Invalid mode: {args.mode}"
    generator = Generator(save_path, entity_dict, args.dataset, device=args.device, cand_size_v2=args.cand_size_v2,
                          topk_v2=args.topk_v2,
                          mode=args.mode)

    test_data = json.load(open(test_data_path))
    test_challenge_data = []
    batch_size = args.batch_size
    n_batch = math.ceil(len(test_data) / batch_size)

    for i in range(n_batch):
        batch_data = test_data[i * batch_size: (i + 1) * batch_size]
        new_batch_data = generator.generate_batch(batch_data)
        test_challenge_data.extend(new_batch_data)
        if i % 100 == 0:
            print(f"{i} / {n_batch} samples have been processed.")

    json.dump(test_challenge_data, open(test_challenge_data_path, 'w'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_root", default="./data", type=str)
    parser.add_argument("--dataset", required=True, type=str, choices=['tacred', 'retacred'])
    parser.add_argument("--ckpt_dir", default="./RE_improved_baseline/ckpts", type=str)
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--cand_size_v2", default=128, type=int)
    parser.add_argument("--topk_v2", default=10, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--mode", default="challenge", type=str, choices=['challenge', 'ibre', 'shuffle'])
    args = parser.parse_args()

    set_seed(args)  # 固定随机数
    main(args)
