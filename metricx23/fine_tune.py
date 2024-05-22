# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""fine-tuning MetricX model."""
'''
1. load dataset
2. load tokenizer, model
3. compute metrics function

'''

import dataclasses
import json
import os
from accelerate import Accelerator
import datasets
from metricx23 import models
from metricx23 import prepare_dataset
import torch
import transformers

from peft import LoraConfig, get_peft_model

from scipy.stats import kendalltau, pearsonr, spearmanr
from peft import prepare_model_for_kbit_training

device_map='auto'


# device_map = {
#         0: [0, 1, 2,3, 4, 5, 6, 7, 8, 9],
#         1: [10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20, 21, 22, 23]
#     }
#device_map = {'': (0, 1)}
#device_map={"": 0,"":1}



print("device ",{'':torch.cuda.current_device()})




@dataclasses.dataclass
class Arguments:
  """Prediction command-line arguments."""

  tokenizer: str = dataclasses.field(
      metadata={"help": "The name of the tokenizer"},
  )

  model_name_or_path: str = dataclasses.field(
      metadata={
          "help": (
              "Path to pretrained model or model identifier from"
              " huggingface.co/models"
          )
      },
  )

  max_input_length: int = dataclasses.field(
      metadata={"help": "The maximum allowable input sequence length."},
  )

  batch_size: int = dataclasses.field(
      metadata={"help": "The global prediction batch size."},
  )

  train_file: str = dataclasses.field(metadata={"help": "The input file."})
  val_file: str = dataclasses.field(metadata={"help": "The input file."})

  output_dir: str = dataclasses.field(
      metadata={"help": "The output file with predictions."},
  )

  qe: bool = dataclasses.field(
      metadata={"help": "Indicates the metric is a QE metric."},
      default=False,
  )
  
# class CustomTrainer(transformers.Trainer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def create_optimizer_and_scheduler(self, num_training_steps):
#         self.optimizer = transformers.AdamW(self.model.parameters(),
#                                lr=self.args.learning_rate,
#                                weight_decay=self.args.weight_decay)
        
#         self.lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
#             self.optimizer, 0, num_training_steps, power=2)
def compute_metrics(pred):
    print("############pred############", pred)
    labels=pred.labels
    # print("labels",labels)
    preds=pred.predictions
    # print("preds", preds)
    k, _ = kendalltau(labels,preds)
    # p, _ = pearsonr(labels,preds)
    # s, _ = spearmanr(labels,preds) {'kendallTau': k}
    return {'kendallTau': k}
        
          
def create_accelerator(gradient_accumulation_steps=1, log_with="wandb"):
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, log_with=log_with)
    if gradient_accumulation_steps > 1:
        accelerator.print("Gradient accumulation steps is", gradient_accumulation_steps)
    return accelerator

    
def main() -> None:

    # Training arguments   
    parser = transformers.HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()
    #print("done line 155")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        per_device_batch_size = args.batch_size // torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        per_device_batch_size = args.batch_size
    ###################################################################################
    # load model and tokenizer   

    bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer , use_fast= False)
    #print("done line 165")
    model = models.MT5ForRegression.from_pretrained(args.model_name_or_path,quantization_config=bnb_config)
    #model.to(device)
    #print(model)

    #model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q","v","k"],
            lora_dropout=0.05,#
            bias="none",
        )
    model = get_peft_model(model, config)
    #model.parallelize(device_map)

    #print("Lora model###############",model)
    model.print_trainable_parameters()
    #print_trainable_parameters(model)

    #model.eval()
    ############################################################################################
    # prepare dataset

    train_obj = prepare_dataset.Prepare_Dataset(
        args.train_file,
        tokenizer,
        args.max_input_length,
        device,
        args.qe, 1
    )
    train_data=train_obj.get_dataset()
    val_obj = prepare_dataset.Prepare_Dataset(
        args.val_file,
        tokenizer,
        args.max_input_length,
        device,
        args.qe, 0
    )
    val_data=val_obj.get_dataset()
    data_collator = transformers.DataCollatorWithPadding(tokenizer)


    #############################################################################
    # Start the training

    logging_step= 50
    #print(ds)
    #os.makedirs(args.output)
    dirname = os.path.dirname(args.output_dir)
    print(dirname)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    training_args = transformers.TrainingArguments(
        output_dir=dirname,
        num_train_epochs=2,
        learning_rate=1e-6,
        per_device_eval_batch_size=per_device_batch_size,
        per_device_train_batch_size=per_device_batch_size,
        evaluation_strategy="epoch",
        logging_steps=logging_step,
        weight_decay=0.01,
        
    )
    # Initialize optimizer and scheduler
    optimizer = transformers.AdamW(model.parameters(), lr=training_args.learning_rate)
    num_training_steps = len(train_data["train"]) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data["train"],
        eval_dataset =val_data["val"],
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler), 
        data_collator=data_collator
    )
    #print("##################", ds['train'])
    #model.config.use_cache=False
    trainer.train()
    
if __name__ == "__main__":
    main()


'''
python metricx23/predict.py --tokenizer google/mt5-xl --model_name_or_path google/metricx-23-xl-v2p0 --max_input_length 1024 --batch_size 1 --input_file metricx/data/test_data.jsonl --output_file metricx/data/test_op.jsonl
export HF_CACHE='/data-3/anushka/.cache'
export HF_HOME='/data-3/anushka/.cache'
export TRANSFORMERS_CACHE='/data-3/anushka/.cache'

CUDA_VISIBLE_DEVICES=0 python -m metricx23.fine_tune \
  --tokenizer google/mt5-xl \
  --model_name_or_path google/metricx-23-xl-v2p0 \
  --max_input_length 1024 \
  --batch_size 32 \
  --train_file data/train_data.jsonl \
  --val_file data/val_data.jsonl \   
  --output_dir ckpts 
'''

# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )  
