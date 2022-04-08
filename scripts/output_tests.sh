echo "STARTING OUTPUT TESTS"
SEED=$1
echo "seed:$SEED"
python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/output_model.json --gpus 0 --use_gpu --seed $SEED
