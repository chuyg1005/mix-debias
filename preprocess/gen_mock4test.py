import argparse
import os
import json


def main(args):
    data_dir = args.data_dir
    dataset = args.dataset
    split = args.split
    mock_ratio = args.mock_ratio

    src_path = os.path.join(data_dir, dataset, f"{split}.json")
    src_data = json.load(open(src_path, "r"))
    save_path = os.path.join(data_dir, dataset, f"{split}_mock.json")

    mock_data = src_data[:int(len(src_data) * mock_ratio)]
    with open(save_path, "w") as f:
        json.dump(mock_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="tacred")
    parser.add_argument("--mock_ratio", type=float, default=0.01)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    main(args)
