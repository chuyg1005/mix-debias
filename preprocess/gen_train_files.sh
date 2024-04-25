# generate train files for training
python preprocess/gen_train_file.py --data_root ./data --dataset tacred --mode aug --k 10 --split train
python preprocess/gen_train_file.py --data_root ./data --dataset retacred --mode aug --k 10 --split train

# generate entity-only files for evaluation
for mode in eo co; do
      for dataset in tacred retacred; do
          for split in dev test; do
              python preprocess/gen_train_file.py --data_root ./data --dataset $dataset --mode $mode --split $split
          done
      done
     for split in dev_rev test_rev; do
         python preprocess/gen_train_file.py --data_root ./data --dataset tacred --mode $mode --split $split
     done
done
