echo "Unsetting env"
unset NCCL_P2P_DISABLE
unset NCCL_IB_DISABLE
unset NCCL_DEBUG
unset NCCL_SOCKET_IFNAME
echo "Starting stuff up."
export SEED=$1
echo "Seed is $SEED"
echo "Starting the process"
CUDA_VISIBLE_DEVICES=1,2 /u/pe25171/anaconda3/envs/kxlnet/bin/python main.py configs/input2_model.json --use_gpu --gpus "2" --train_batch_size 16 --epochs 2 --seed $SEED --cache_dir "./cache_$SEED" &
CUDA_VISIBLE_DEVICES=3,5 /u/pe25171/anaconda3/envs/kxlnet/bin/python main.py configs/output_claire_model.json --use_gpu --gpus "2" --train_batch_size 16 --fp16 --epochs 2 --seed $SEED --cache_dir "./cache_$SEED" &
#CUDA_VISIBLE_DEVICES=4,5 /u/pe25171/anaconda3/envs/kxlnet/bin/python main.py configs/output_claire_model_300k.json --use_gpu --gpus "2" --train_batch_size 16 --fp16 --epochs 2 --seed $SEED --cache_dir "./cache_$SEED"
CUDA_VISIBLE_DEVICES=6,7 /u/pe25171/anaconda3/envs/kxlnet/bin/python main.py configs/arch_model.json --use_gpu --gpus "2" --train_batch_size 16 --epochs 2 --seed $SEED --cache_dir "./cache_$SEED"


#CUDA_VISIBLE_DEVICES=6,7 /u/pe25171/anaconda3/envs/kxlnet/bin/python main.py configs/output_claire_model_600k.json --use_gpu --gpus "2" --train_batch_size 16 --fp16 --epochs 2 --seed $SEED --cache_dir "./cache_$SEED"
