CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/transfg/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir exp_00/debug1 --dataset CUB_200_2011 --train_batch_size 16 --workers 4 --arch_type TransFG --model_type ViT-B_16 --pretrained_dir logs/pretrained_ViT/imagenet21k_ViT-B_16.npz --split non-overlap --num_steps 10000 --eval_every 200 --fp16 --name sample_run --round 1 &


######### run01
# non-overlap
hfai python tools/exp00_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp00_v02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split non-overlap --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_nonoverlap_r1 &\

hfai python tools/exp00_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp00_v02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split non-overlap --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_nonoverlap_r2 &\

hfai python tools/exp00_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp00_v02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split non-overlap --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_nonoverlap_r3 &\

# overlap
hfai python tools/exp00_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp00_v02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_overlap_r1 &\

hfai python tools/exp00_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp00_v02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_overlap_r2 &\

hfai python tools/exp00_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp00_v02/run01 \
--model_type ViT-B_16 --pretrained_dir pretrained_ViT/imagenet21k_ViT-B_16.npz \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--split overlap --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_overlap_r3 &\

# resnet50
hfai python tools/exp00_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp00_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_r50_r1 &\

hfai python tools/exp00_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp00_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_r50_r2 &\

hfai python tools/exp00_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp00_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_r50_r3 &\

######### run02 (short epoch)
# 20 epoch
python tools/exp00_v02/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp00_v02/run02 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 4 \
--epochs 20 --warmup_epochs 5 --round 1 &&\

python tools/exp00_v02/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp00_v02/run02 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 4 \
--epochs 20 --warmup_epochs 5 --round 2 &&\

python tools/exp00_v02/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp00_v02/run02 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 4 \
--epochs 20 --warmup_epochs 5 --round 3 &&\

# 30 epoch
python tools/exp00_v02/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp00_v02/run02 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 4 \
--epochs 30 --warmup_epochs 5 --round 1 &&\

python tools/exp00_v02/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp00_v02/run02 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 4 \
--epochs 30 --warmup_epochs 5 --round 2 &&\

python tools/exp00_v02/train_local.py \
--output_dir_root /root/share/TransFG/ --output_dir output/exp00_v02/run02 \
--model_type resnet50 \
--data_root /root/data/public \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 4 \
--epochs 30 --warmup_epochs 5 --round 3 &&\