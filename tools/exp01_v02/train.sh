# debug
hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01_debug2 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 8 --lr 3e-2 --workers 1 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 1.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_bsl_e01v02_lbd1x0_r1 &\



# run01
hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.1 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x1_r1 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.1 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x1_r2 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.1 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x1_r3 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.1 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x1_r4 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.1 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x1_r5 &\




hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.3 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x3_r1 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.3 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x3_r2 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.3 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x3_r3 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.3 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x3_r4 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.3 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x3_r5 &\




hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.5 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x5_r1 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.5 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x5_r2 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.5 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x5_r3 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.5 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x5_r4 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.5 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x5_r5 &\




hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.7 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x7_r1 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.7 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x7_r2 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.7 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x7_r3 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.7 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x7_r4 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 0.7 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd0x7_r5 &\




hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 1.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd1x0_r1 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 1.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd1x0_r2 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 1.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd1x0_r3 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 1.0 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd1x0_r4 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 1.0 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd1x0_r5 &\




hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 3.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd3x0_r1 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 3.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd3x0_r2 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 3.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd3x0_r3 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 3.0 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd3x0_r4 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 3.0 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd3x0_r5 &\




hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 5.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd5x0_r1 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 5.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd5x0_r2 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 5.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd5x0_r3 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 5.0 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd5x0_r4 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 5.0 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd5x0_r5 &\




hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 7.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd7x0_r1 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 7.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd7x0_r2 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 7.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd7x0_r3 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 7.0 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd7x0_r4 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 7.0 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd7x0_r5 &\




hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 7.0 --round 1 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd9x0_r1 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 7.0 --round 2 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd9x0_r2 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 7.0 --round 3 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd9x0_r3 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 7.0 --round 4 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd9x0_r4 &\

hfai python tools/exp01_v02/train.py \
--output_dir_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/TransFG --output_dir output/exp01_v02/run01 \
--model_type resnet50 \
--data_root /ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets \
--dataset CUB_200_2011 --train_batch_size 64 --lr 3e-2 --workers 8 \
--epochs 100 --warmup_epochs 5 \
--lambda_0 9.0 --round 5 \
-- --nodes=1 --priority=10 --name=pyf_CUB_r50_e01v02_lbd9x0_r5 &\