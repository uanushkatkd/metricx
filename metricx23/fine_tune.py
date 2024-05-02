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

import datasets
from metricx23 import models
import torch
import transformers

from peft import LoraConfig, get_peft_model

from scipy.stats import kendalltau, pearsonr, spearmanr
from peft import prepare_model_for_kbit_training






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

  input_file: str = dataclasses.field(metadata={"help": "The input file."})

  output_file: str = dataclasses.field(
      metadata={"help": "The output file with predictions."},
  )

  qe: bool = dataclasses.field(
      metadata={"help": "Indicates the metric is a QE metric."},
      default=False,
  )

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )  


def get_dataset(
    input_file: str, tokenizer, max_input_length: int, device, is_qe: bool
):
  """Gets the test dataset for prediction.

  If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
  If it is false, there must be "hypothesis" and "reference" fields.

  Args:
    input_file: The path to the jsonl input file.
    tokenizer: The tokenizer to use.
    max_input_length: The maximum input sequence length.
    device: The ID of the device to put the PyTorch tensors on.
    is_qe: Indicates whether the metric is a QE metric or not.

  Returns:
    The dataset.
  """
  
  def _make_input(example):
      if is_qe:
          example["input"] = (
              "candidate: "
              + example["hypothesis"]
              + " source: "
              + example["source"]
          )
      else:
          example["input"] = (
              "candidate: "
              + example["hypothesis"]
              + " reference: "
              + example["reference"]
          )
      return example

  def _tokenize(example):
      return tokenizer(
          example["input"],
          max_length=max_input_length,
          truncation=True,
          padding=False, 
      )

  def _remove_eos(example):
      example["input_ids"] = example["input_ids"][:-1]
      example["attention_mask"] = example["attention_mask"][:-1]
      return example

  def _pad(example):
      #print("################### in")
      input_ids = example["input_ids"]
      attention_mask = example["attention_mask"]
      padded_input_ids = input_ids + [tokenizer.pad_token_id] * (max_input_length - len(input_ids))
      padded_attention_mask = attention_mask + [0] * (max_input_length - len(attention_mask))
      example["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
      example["attention_mask"] = torch.tensor(padded_attention_mask, dtype=torch.long)
      #print(len(example["input_ids"]))
      #print("################## out")
      return example

  ds = datasets.load_dataset("json", data_files={"train": input_file})
  print(ds)
  print("##################yes ds")
  ds = ds.map(_make_input)
  ds = ds.map(_tokenize)
  ds = ds.map(_remove_eos)
  ds = ds.map(_pad)
  print("out ds")
  ds.set_format(
      type="torch",
      columns=["input_ids", "attention_mask"],
      device=device,
      output_all_columns=True,
  )
  print(ds)
  return ds
def compute_metrics(pred):
    print("%%%%%%%%%%%%%%%%%%%%%%%%%", pred)
    labels=pred.label_ids
    print("labels",labels)
    preds=pred.predictions
    print("preds", preds)
    k, _ = kendalltau(labels,preds)
    p, _ = pearsonr(labels,preds)
    s, _ = spearmanr(labels,preds)
    return {'kendalltau':k,'pearsonr':p,'spearman':s}
    

def main() -> None:
  parser = transformers.HfArgumentParser(Arguments)
  (args,) = parser.parse_args_into_dataclasses()
  print("done line 155")

  if torch.cuda.is_available():
    device = torch.device("cuda")
    per_device_batch_size = args.batch_size // torch.cuda.device_count()
  else:
    device = torch.device("cpu")
    per_device_batch_size = args.batch_size
    
    
  bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer , use_fast= False)
  print("done line 165")


  model = models.MT5ForRegression.from_pretrained(args.model_name_or_path, device_map="auto")
  #model.to(device)
  #print(model)
  
  
  model.gradient_checkpointing_enable()
  #model = prepare_model_for_kbit_training(model)
  config = LoraConfig(
          r=8,
          lora_alpha=8,
          target_modules=["q","v"],
          lora_dropout=0.5,
          bias="none",
          task_type='SEQ_2_SEQ_LM',
      )
  #model = get_peft_model(model, config)

  #print(model)
  print_trainable_parameters(model)
  
  #model.eval()

  ds = get_dataset(
      args.input_file,
      tokenizer,
      args.max_input_length,
      device,
      args.qe,
  )
  print("################ dataset",ds)
  ds=ds.rename_column("mqm","labels")
  logging_step= len(ds["train"]) //per_device_batch_size
  print(ds)
  
  
  training_args = transformers.TrainingArguments(
      output_dir=os.path.dirname(args.output_file),
      num_train_epochs=1,
      learning_rate=2e-5,
      per_device_eval_batch_size=per_device_batch_size,
      per_device_train_batch_size=per_device_batch_size,
      evaluation_strategy="epoch",
      logging_steps=logging_step,
      fp16=True,
      weight_decay=0.01,
  )
  
  trainer = transformers.Trainer(
      model=model,
      args=training_args,
      compute_metrics=compute_metrics,
      train_dataset=ds["train"],
      tokenizer=tokenizer
  )
  print("##################", ds['train'])

  trainer.train()
'''dirname = os.path.dirname(args.output_file)
  if dirname:
    os.makedirs(dirname, exist_ok=True)

  with open(args.output_file, "w", encoding='utf-8') as out:
    for pred, example in zip(predictions, ds["test"]):
      example["prediction"] = float(pred)
      del example["input"]
      del example["input_ids"]
      del example["attention_mask"]
      out.write(json.dumps(example , ensure_ascii = False) + "\n")

   '''
if __name__ == "__main__":
    main()


'''
python metricx23/predict.py --tokenizer google/mt5-xl --model_name_or_path google/metricx-23-xl-v2p0 --max_input_length 1024 --batch_size 1 --input_file metricx/data/test_data.jsonl --output_file metricx/data/test_op.jsonl
export HF_CACHE='/data-3/anushka/.cache'
export HF_HOME='/data-3/anushka/.cache'
export TRANSFORMERS_CACHE='/data-3/anushka/.cache'

python -m metricx23.predict \
  --tokenizer google/mt5-xl \
  --model_name_or_path google/metricx-23-xl-v2p0 \
  --max_input_length 1024 \
  --batch_size 16 \
  --input_file data/train_data.jsonl \
  --output_file data/train_op.jsonl
'''