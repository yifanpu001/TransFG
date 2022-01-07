CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/transfg/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir exp_00/debug1 --dataset CUB_200_2011 --train_batch_size 16 --workers 4 --arch_type TransFG --model_type ViT-B_16 --pretrained_dir logs/pretrained_ViT/imagenet21k_ViT-B_16.npz --split non-overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 1 &


hfai python train_ddp.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp_00/debug2 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --workers 4 \
--arch_type Pure --model_type ViT-B_16 --pretrained_dir logs/pretrained_ViT/imagenet21k_ViT-B_16.npz \
--split overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 1 \
-- --nodes=1 --priority=40 --name=pyffg_debug1 &\