#echo "Unsetting env"
#unset NCCL_P2P_DISABLE
#unset NCCL_IB_DISABLE
#unset NCCL_DEBUG
#unset NCCL_SOCKET_IFNAME
echo "Starting stuff up."
export SEED=$1
echo "Seed is $SEED"
echo "Starting the process"
CUDA_VISIBLE_DEVICES=2,3 /u/pe25171/anaconda3/envs/kxlneta6000/bin/python main.py configs/base_model.json --use_gpu --gpus "2" --fp16 --train_batch_size 16 --epochs 2 --seed $SEED --cache_dir "./cache_$SEED" &
CUDA_VISIBLE_DEVICES=2,3 /u/pe25171/anaconda3/envs/kxlneta6000/bin/python main.py configs/output_claire_model_learnable.json --use_gpu --train_batch_size 16 --gpus "2" --epochs 2 --seed $SEED --cache_dir "./cache_$SEED"
CUDA_VISIBLE_DEVICES=2,3 /u/pe25171/anaconda3/envs/kxlneta6000/bin/python main.py configs/input2_model_sum.json --use_gpu --gpus "2" --train_batch_size 16 --epochs 2 --seed $SEED --cache_dir "./cache_$SEED" &
CUDA_VISIBLE_DEVICES=2,3 /u/pe25171/anaconda3/envs/kxlneta6000/bin/python main.py configs/output2_model.json --use_gpu --gpus "2" --fp16 --train_batch_size 16 --epochs 2 --seed $SEED --cache_dir "./cache_$SEED"
