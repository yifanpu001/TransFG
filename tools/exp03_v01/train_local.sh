# resnet50 debug

# for slurm
# for local server
# --output_dir_root /home/pyf/logs/fg_log --output_dir output/exp03_v01/debug01 \
# for jiutian

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/exp03_v01/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp03_v01/debug04 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 128 --lr 3e-2 --workers 4 \
--epochs 8 --warmup_epochs 5 --round 1 \
--meta_lr 3e-2 --meta_net_hidden_size 512 --meta_net_num_layers 1 --meta_weight_decay 0.0 --lambda_0 1.0 --debug_flag 1 &\


# resnet50 run
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/exp03_v01/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp03_v01/run01 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 128 --lr 3e-2 --workers 4 \
--epochs 20 --warmup_epochs 5 --round 1 \
--meta_lr 3e-1 --meta_net_hidden_size 512 --meta_net_num_layers 1 --meta_weight_decay 0.0 --lambda_0 1.0 --debug_flag 1 &&\

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/exp03_v01/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp03_v01/run01 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 128 --lr 3e-2 --workers 4 \
--epochs 20 --warmup_epochs 5 --round 1 \
--meta_lr 3e-2 --meta_net_hidden_size 512 --meta_net_num_layers 1 --meta_weight_decay 0.0 --lambda_0 1.0 --debug_flag 1 &&\

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/exp03_v01/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp03_v01/run01 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 128 --lr 3e-2 --workers 4 \
--epochs 20 --warmup_epochs 5 --round 1 \
--meta_lr 3e-3 --meta_net_hidden_size 512 --meta_net_num_layers 1 --meta_weight_decay 0.0 --lambda_0 1.0 --debug_flag 1 &&\

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/exp03_v01/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp03_v01/run01 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 128 --lr 3e-2 --workers 4 \
--epochs 20 --warmup_epochs 5 --round 1 \
--meta_lr 3e-4 --meta_net_hidden_size 512 --meta_net_num_layers 1 --meta_weight_decay 0.0 --lambda_0 1.0 --debug_flag 1 &\
