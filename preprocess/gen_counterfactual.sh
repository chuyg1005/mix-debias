# generate challenge files by counterfauctal filtering
# require pregenerated probs of default model under entity-only setting

dataset=$1

for model_name in "ibre" "luke"; do
    for split in "test" "dev"; do
      python preprocess/filter_counterfactual_samples.py --dataset $dataset --split $split --model_name $model_name
    done
    if [ "$dataset" == "tacred" ]; then
      python preprocess/filter_counterfactual_samples.py --dataset tacred --split dev_rev --model_name $model_name
      python preprocess/filter_counterfactual_samples.py --dataset tacred --split test_rev --model_name $model_name
    fi
done
