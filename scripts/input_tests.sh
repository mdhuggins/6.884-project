echo "STARTING INPUT TESTS"
SEED=$1
echo "seed:$SEED"
python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/input_model.json --gpus 2 --use_gpu --seed $SEED &
