# debug
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/debug01/p30 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.12 --round 1 \
-- --nodes=1 --priority=30 --name=pyf_CUB_e00_lbd000x12_r1_30debug &\


# run01
## lbd=000.10
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.1 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x10_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.1 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x10_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.1 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x10_r3 &\

## lbd=000.50
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.5 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x50_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.5 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x50_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.5 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x50_r3 &\

## lbd=001.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 1.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd001x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 1.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd001x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 1.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd001x00_r3 &\

## lbd=003.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 3.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd003x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 3.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd003x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 3.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd003x00_r3 &\

## lbd=005.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 5.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd005x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 5.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd005x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 5.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd005x00_r3 &\


## lbd=007.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 7.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd007x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 7.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd007x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 7.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd007x00_r3 &\


## lbd=009.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 9.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd009x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 9.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd009x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 9.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd009x00_r3 &\












# run02
## lbd=000.10
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.1 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x10_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.1 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x10_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.1 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x10_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.1 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x10_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.1 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x10_r5 &\

## lbd=000.50
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.5 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x50_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.5 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x50_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.5 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x50_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.5 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x50_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.5 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd000x50_r5 &\


## lbd=001.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 1.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd001x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 1.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd001x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 1.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd001x00_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 1.0 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd001x00_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 1.0 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd001x00_r5 &\

## lbd=003.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 3.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd003x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 3.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd003x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 3.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd003x00_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 3.0 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd003x00_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 3.0 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd003x00_r5 &\

## lbd=005.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 5.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd005x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 5.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd005x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 5.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd005x00_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 5.0 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd005x00_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 5.0 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd005x00_r5 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 5.0 --round 6 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd005x00_r6 &\

## lbd=007.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 7.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd007x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 7.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd007x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 7.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd007x00_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 7.0 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd007x00_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 7.0 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd007x00_r5 &\


## lbd=009.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 9.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd009x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 9.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd009x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 9.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd009x00_r3 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 9.0 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd009x00_r4 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 9.0 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd009x00_r5 &\


## lbd=002.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 2.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd002x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 2.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd002x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 2.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd002x00_r3 &\

## lbd=004.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 4.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd004x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 4.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd004x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 4.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd004x00_r3 &\

## lbd=006.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 6.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd006x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 6.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd006x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 6.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd006x00_r3 &\


## lbd=008.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 8.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd008x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 8.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd008x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 8.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd008x00_r3 &\


## lbd=010.00
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 10.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd010x00_r1 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 10.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd010x00_r2 &\

hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/run02 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 10.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e01_lbd010x00_r3 &\

