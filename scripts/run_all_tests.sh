echo "STARTING TESTS"
cd ..
for i in $(seq 1 10);
do
    SEED=$RANDOM
    echo "ITERATION: $i SEED: $SEED"
    python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/base_model.json --gpus 0 --use_gpu --seed $SEED &
    python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/input_model.json --gpus 1 --use_gpu --seed $SEED &
    python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/kadapt-10-11.json --gpus 2 --use_gpu --seed $SEED &
    python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/kadapt-10-11-concat.json --gpus 3 --use_gpu --seed $SEED
#    python main.py --epochs 3 --num_workers 2 --config_file configs/kadapt-10-11-noinject.json --gpus 0 --use_gpu --seed $SEED &
    python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/kadapt-11.json --gpus 1 --use_gpu --seed $SEED &
    python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/kadapt-all.json --gpus 2 --use_gpu --seed $SEED &
    python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/kadapt-none.json --gpus 3 --use_gpu --seed $SEED &
    python main.py --epochs 3 --num_workers 2 --train_batch_size 32 --config_file configs/output_model.json --gpus 0 --use_gpu --seed $SEED
    echo "Removing Cache!!"
    rm -rf ./cache
    echo "Done"
done
echo"Done"