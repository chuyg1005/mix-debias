dataset=$1
model_name=$2
mode=$3
seed=$4
dev_name=$5
test_name=$6
python core_cli.py \
  --dataset $dataset \
  --model_name $model_name \
  --mode $mode \
  --seed $seed \
  --dev_name $dev_name \
  --test_name $test_name
