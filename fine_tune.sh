export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
    --num_machines 1 \
    --num_processes 2 \
    /data/anushka/metricx/metricx23/fine_tune_from_scratch.py \
    --tokenizer google/mt5-large \
    --model google/metricx-23-qe-large-v2p0 \
    --gradient_accumulation_steps 8 \
    --max_input_length 1024 \
    --batch_size 16 \
    --train_file /data/anushka/metricx/wmtDataset/transliterated_train.jsonl \
    --val_file /data/anushka/metricx/wmtDataset/transliterated_dev.jsonl \
    --output_file /data/anushka/metricx/ckpts/
