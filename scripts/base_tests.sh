echo "STARTING BASE TESTS"
SEED=$1
echo "seed:$SEED"
python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/base_model.json --gpus 1 --use_gpu --seed $SEED &
