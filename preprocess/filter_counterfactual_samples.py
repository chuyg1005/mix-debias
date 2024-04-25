import argparse
import numpy as np
import os
import json


def main(args):
    data_dir = args.data_dir
    model_name = args.model_name
    mode = args.mode
    seed = args.seed
    dataset = args.dataset
    split = args.split

    src_path = os.path.join(data_dir, dataset, f"{split}.json")
    src_data = json.load(open(src_path, "r"))
    save_path = os.path.join(data_dir, dataset, f"{split}_{model_name}.json")

    preds_path = os.path.join(data_dir, "predictions", model_name, dataset, f"{mode}-{seed}", f"{split}-eo.txt")
    if not os.path.exists(preds_path):
        print(f"Model prediction file {preds_path} not exists.")
        return
    preds = np.loadtxt(preds_path, dtype=float, delimiter=',')

    labels = preds[:, -1].astype(int)
    probs = preds[:, :-1]

    # 筛选出challenge_set
    predictions = np.argmax(probs, axis=-1)
    challenge_set = np.where(predictions != labels)[0]
    challenge_data = [src_data[i] for i in challenge_set]

    # 保存challenge_data
    print(
        f"model: {model_name}, mode: {mode}, seed: {seed}, dataset: {dataset}, split: {split}, #challenge_data: {len(challenge_data)}")
    with open(save_path, "w") as f:
        json.dump(challenge_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_name", type=str, default="ibre")
    parser.add_argument("--mode", type=str, default="default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="tacred")
    parser.add_argument("--split", type=str, default="test")

    args = parser.parse_args()
    main(args)
