echo "Unsetting env"
unset NCCL_P2P_DISABLE
unset NCCL_IB_DISABLE
unset NCCL_DEBUG
unset NCCL_SOCKET_IFNAME
echo "Starting stuff up."
/u/pe25171/anaconda3/envs/kxlnet/bin/python main.py configs/base_model.json --use_gpu --gpus 0,1 --fp16 --epochs 2 &
/u/pe25171/anaconda3/envs/kxlnet/bin/python main.py configs/output2_model.json --use_gpu --gpus 2,3 --fp16 --epochs 2 &
/u/pe25171/anaconda3/envs/kxlnet/bin/python main.py configs/output_claire_model.json --use_gpu --gpus 4,5 --fp16 --epochs 2 &
/u/pe25171/anaconda3/envs/kxlnet/bin/python main.py configs/output_claire_model_300k.json --use_gpu --gpus 6,7 --fp16 --epochs 2
/u/pe25171/anaconda3/envs/kxlnet/bin/python main.py configs/output_claire_model_600k.json --use_gpu --gpus 6,7 --fp16 --epochs 2
