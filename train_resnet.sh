# CUDA_VISIBLE_DEVICES=4,5 python train_resnet.py -a resnet50 \
# --epochs 90 --batch-size 512 --lr 0.4 --workers 16 \
# --dist-url 'tcp://127.0.0.1:12000' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /home/data/imagenet &

CUDA_VISIBLE_DEVICES=4,5 python train_resnet_cub.py -a resnet18 --train_batch_size 64 --eval_batch_size 64 --lr 0.4 --workers 4 \
--dataset CUB_200_2011 --data_root /cluster/home2/pyf_log/datasets/fine_grained/CUB_200_2011 --dist-url 'tcp://127.0.0.1:12002' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --print-freq 20 /home/data/imagenet &