# debug
hfai python train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01/debug01 \
--model_type ViT-B_16 --pretrained_dir logs/pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --lambda_0 0.12 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_e00_lbd000x12_r1_debug &\