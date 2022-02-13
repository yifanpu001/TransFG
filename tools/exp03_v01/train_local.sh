# resnet50 debug
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python tools/exp03_v01/train_local.py \
--output_dir_root /home/pyf/logs/fg_log --output_dir output/exp03_v01/debug01 \
--model_type resnet50 \
--data_root /home/pyf/data \
--dataset CUB_200_2011 --train_batch_size 16 --lr 3e-2 --workers 1 \
--epochs 1 --warmup_epochs 5 --round 1 \
--meta_lr 3e-2 --meta_net_hidden_size 512 --meta_net_num_layers 1 --meta_weight_decay 0.0 --lambda_0 1.0 --debug_flag 1 &\