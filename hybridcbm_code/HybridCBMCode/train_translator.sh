export NUM_NODES=1
export NUM_GPUS_PER_NODE=2  # Đã sửa: Kaggle chỉ có tối đa 2 GPU
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355

# 2. Gọi 2 con GPU số 0 và số 1 của Kaggle
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$NODE_RANK \
    --rdzv_id=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_translator.py \
    --out_dir /kaggle/working/weights/translator-AddData \
    --epochs 100 \
    --save_every 2 \
    --batch_size 128 \
    --clip_model ViT-L/14 \
    --augment