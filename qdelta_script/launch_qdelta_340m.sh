#!/usr/bin/bash

echo "Clearing Triton cache to force recompilation with smaller block sizes..."
rm -rf /tmp/triton_cache ~/.triton/cache ~/.triton 2>/dev/null || true
mkdir -p /tmp/triton_cache

source blackwell_setup.sh

echo "Triton optimization settings applied:"
echo "  - Num stages: ${TRITON_DEFAULT_NUM_STAGES:-1}"
echo "  - Num warps: ${TRITON_DEFAULT_NUM_WARPS:-2}"
echo "  - Max shared mem: ${TRITON_MAX_SHARED_MEMORY:-98304} bytes"
echo ""

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} NGPU=${NGPU:-4} bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder exp/qdelta-340M-4K-15B/qdelta.batch1.seqlen65536.context4096.warmup1024.update2.steps28600.lr1e-3.cosine \
  --model.config configs/qdelta_340M.json \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --optimizer.name AdamW \
  --optimizer.eps 1e-15 \
  --optimizer.lr 1e-3 \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --lr_scheduler.decay_type cosine \
  --training.batch_size 1 \
  --training.seq_len 65536 \
  --training.context_len 4096 \
  --training.data_parallel_shard_degree 4 \
  --training.gradient_accumulation_steps 2 \
  --training.steps 28600 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.dataset HuggingFaceFW/fineweb-edu \
  --training.dataset_name sample-100BT \
  --training.dataset_split train \
  --training.num_workers 32 \
  --training.prefetch_factor 2 \
  --training.mixed_precision_param bfloat16 \
  --training.seed 42 \
  --training.compile \
  --checkpoint.interval 5000 \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 2 \
  --metrics.log_freq 1 

