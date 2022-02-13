# resnet50 debug
hfai python tools/exp03_v01/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp03_v01/debug09 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 16 --lr 3e-2 --workers 4 \
--epochs 100 --warmup_epochs 5 --round 1 \
--meta_lr 3e-2 --meta_net_hidden_size 512 --meta_net_num_layers 1 --meta_weight_decay 0.0 --lambda_0 1.0 \
-- --nodes=1 --priority=30 --name=pyf_CUB_r50_e03v01_de &\