python -m metricx23.predict \
  --tokenizer google/mt5-xl \
  --model_name_or_path google/metricx-23-qe-xl-v2p0 \
  --max_input_length 1024 \
  --batch_size 32 \
  --input_file wmtDataset/wmt_dev.jsonl \
  --output_file wmtDataset/wmt_dev_metricx.jsonl

