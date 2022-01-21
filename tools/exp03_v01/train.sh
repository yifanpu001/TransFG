CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/transfg/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir exp_00/debug1 --dataset CUB_200_2011 --train_batch_size 16 --workers 4 --arch_type TransFG --model_type ViT-B_16 --pretrained_dir logs/pretrained_ViT/imagenet21k_ViT-B_16.npz --split non-overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 1 &


# resnet50 debug
hfai python tools/exp03_v01/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp03_v01/debug01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 16 --lr 3e-2 --workers 2 \
--epochs 100 --warmup_epochs 5 --round 1 \
--meta_lr 3e-2 --lambda_0 1.0 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_r50_r1 &\