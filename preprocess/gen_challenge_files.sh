device=$1
dataset=$2
batch_size=$3
export CUDA_VISIBLE_DEVICES=$device

for mode in "challenge" "shuffle"; do
    for split in "dev" "test"; do
      python preprocess/gen_challenge_file.py --dataset "$dataset" --mode $mode --batch_size "$batch_size" --split $split
    done
    if [ "$dataset" == "tacred" ]; then
      # shellcheck disable=SC2086
      python preprocess/gen_challenge_file.py --dataset "$dataset" --mode $mode --batch_size "$batch_size" --split dev_rev
      python preprocess/gen_challenge_file.py --dataset "$dataset" --mode $mode --batch_size "$batch_size" --split test_rev
    fi
done
