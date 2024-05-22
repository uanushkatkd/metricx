CUDA_VISIBLE_DEVICES=0,1 python -m metricx23.fine_tune \
  --tokenizer  google/mt5-large \
  --model_name_or_path google/metricx-23-large-v2p0 \
  --max_input_length 1024 \
  --batch_size 64 \
  --train_file data/train_data.jsonl \
  --val_file data/val_data.jsonl \
  --output_dir ./ckpts
