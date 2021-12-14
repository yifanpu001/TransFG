CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 train.py \
--output_dir /cluster/home/pyf/code/FGVA/TransFG/output/debug2 \
--dataset CUB_200_2011 --train_batch_size 4 \
--model_type ViT-B_16 --pretrained_dir /home/pyf_log/TransFG_log/pretrained_ViT/ViT-B_16.npz \
--split overlap --num_steps 40000 --eval_every 1000 \
--fp16 --name sample_run &