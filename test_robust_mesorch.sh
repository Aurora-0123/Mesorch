base_dir="./eval_robust_dir_mesorch"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=0 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./test_robust.py \
    --model Mesorch \
    --world_size 1 \
    --test_data_path "/root/autodl-tmp/Mesorch/public_datasets/IML/casia" \
    --checkpoint_path "./ckpt_mesorch/mesorch-98.pth" \
    --test_batch_size 2 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --seed 42 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log