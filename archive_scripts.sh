CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/transfg/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir exp00/baseline_v1 --dataset CUB_200_2011 --train_batch_size 16 --workers 4 --arch_type Pure --model_type ViT-B_16 --pretrained_dir /root/share/TransFG/logs/pretrained_ViT/imagenet21k_ViT-B_16.npz --split overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 1 &&
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/transfg/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir exp00/baseline_v1 --dataset CUB_200_2011 --train_batch_size 16 --workers 4 --arch_type Pure --model_type ViT-B_16 --pretrained_dir /root/share/TransFG/logs/pretrained_ViT/imagenet21k_ViT-B_16.npz --split overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 2 &&
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/transfg/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir exp00/baseline_v1 --dataset CUB_200_2011 --train_batch_size 16 --workers 4 --arch_type Pure --model_type ViT-B_16 --pretrained_dir /root/share/TransFG/logs/pretrained_ViT/imagenet21k_ViT-B_16.npz --split overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 3 &

CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/transfg/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir exp00/baseline_v2 --dataset CUB_200_2011 --train_batch_size 16 --workers 4 --arch_type Pure --model_type ViT-B_16 --pretrained_dir /root/share/TransFG/logs/pretrained_ViT/imagenet21k_ViT-B_16.npz --split non-overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 1 &&
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/transfg/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir exp00/baseline_v2 --dataset CUB_200_2011 --train_batch_size 16 --workers 4 --arch_type Pure --model_type ViT-B_16 --pretrained_dir /root/share/TransFG/logs/pretrained_ViT/imagenet21k_ViT-B_16.npz --split non-overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 2 &&
CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/transfg/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir exp00/baseline_v2 --dataset CUB_200_2011 --train_batch_size 16 --workers 4 --arch_type Pure --model_type ViT-B_16 --pretrained_dir /root/share/TransFG/logs/pretrained_ViT/imagenet21k_ViT-B_16.npz --split non-overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 3 &