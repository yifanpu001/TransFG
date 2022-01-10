CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/transfg/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir exp_00/debug1 --dataset CUB_200_2011 --train_batch_size 16 --workers 4 --arch_type TransFG --model_type ViT-B_16 --pretrained_dir logs/pretrained_ViT/imagenet21k_ViT-B_16.npz --split non-overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 1 &


######### run01
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.001 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.001_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.001 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.001_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.001 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.001_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.001 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.001_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.001 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.001_r5 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.001 --round 6 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.001_r6 &\






hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.003 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.003_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.003 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.003_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.003 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.003_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.003 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.003_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.003 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.003_r5 &\







hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.005 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.005_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.005 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.005_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.005 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.005_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.005 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.005_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.005 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.005_r5 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.005 --round 6 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.005_r6 &\








hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.007 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.007_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.007 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.007_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.007 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.007_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.007 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.007_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.007 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.007_r5 &\







hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.01 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.01_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.01 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.01_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.01 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.01_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.01 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.01_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.01 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.01_r5 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.01 --round 6 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.01_r6 &\






hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.03 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.03_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.03 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.03_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.03 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.03_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.03 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.03_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.03 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.03_r5 &\




hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.05 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.05_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.05 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.05_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.05 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.05_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.05 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.05_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.05 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.05_r5 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.05 --round 6 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.05_r6 &\





hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.07 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.07_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.07 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.07_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.07 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.07_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.07 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.07_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.07 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.07_r5 &\



hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.1 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.1_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.1 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.1_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.1 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.1_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.1 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.1_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.1 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.1_r5 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.1 --round 6 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.1_r6 &\






hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.3 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.3_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.3 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.3_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.3 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.3_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.3 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.3_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.3 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.3_r5 &\







hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.5 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.5_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.5 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.5_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.5 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.5_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.5 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.5_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.5 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.5_r5 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.5 --round 6 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.5_r6 &\






hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.7 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.7_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.7 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.7_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.7 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.7_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.7 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.7_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --loss_type ce_me --gamma 0.7 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_exp02_overlap_g0.7_r5 &\
