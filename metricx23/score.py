import dataclasses
import argparse
import json
from tqdm import tqdm
import os
from accelerate import Accelerator
import datasets
from metricx23 import models
import torch
import transformers
import wandb
from torch.utils.data import DataLoader
import pandas as pd
from peft import PeftConfig, PeftModel
import bitsandbytes as bnb

from bitsandbytes.functional import dequantize_4bit
from peft.utils import _get_submodules
import copy


from scipy.stats import kendalltau, pearsonr, spearmanr

class Prepare_Dataset:
    def __init__(self, file_path: str, tokenizer, max_input_length: int, device, is_qe: bool) -> None:
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.device = device
        self.is_qe = is_qe

    def _make_input(self, example):
        hypothesis = example.get("hypothesis", "")
        if self.is_qe:
            source = example.get("source", "")
            example["input"] = f"candidate: {hypothesis} source: {source}"
        else:
            reference = example.get("reference", "")
            example["input"] = f"candidate: {hypothesis} reference: {reference}"
        return example

    def _tokenize(self, example):
        return self.tokenizer(
            example["input"],
            max_length=self.max_input_length,
            padding=True, 
            truncation=True
        )

    def _remove_eos(self, example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    def map_dataset(self, ds):
        ds = ds.map(self._make_input)
        ds = ds.map(self._tokenize)
        ds = ds.map(self._remove_eos)
        #ds = ds.rename_column("mqm", "labels")
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.device
        )
        return ds

    def get_dataset(self):          
        dataset = datasets.load_dataset("json", data_files={"test": self.file_path})['test']
        mapped_dataset = self.map_dataset(dataset)
        return mapped_dataset

    def create_dataloader(self, batch_size):
        dataset = self.get_dataset()
        data_collator = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator, num_workers=0)
        return dataloader



def load_checkpoint(ckpt, model_name):
    # Load model and tokenizer from checkpoint directory
    model =  models.MT5ForRegression.from_pretrained(
            model_name ,
            torch_dtype=torch.bfloat16
        )
    #base_model = dequantize_model(base_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt , use_fast= False)
    
    return model, tokenizer


def evaluate_model(model, tokenizer, test_dataloader, device):
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    for batch in test_dataloader:
        with torch.no_grad():
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            print(outputs)
            predictions = outputs.predictions
            all_predictions.extend(predictions.cpu().float().numpy())
            #all_labels.extend(batch["labels"].cpu().numpy())
    
    # Calculate metrics (e.g., Kendalltau score)
    #kendalltau_score = compute_metrics(all_predictions, all_labels)
    #print(f"Kendalltau Score: {kendalltau_score}")
    return all_predictions
def compute_metrics(pred,labels):
    #print("############pred############", pred)
    #labels=pred.labels
    # print("labels",labels)
    #preds=pred.predictions
    # print("preds", preds)
    k, _ = kendalltau(labels,pred)
    p, _ = pearsonr(labels,pred)
    s, _ = spearmanr(labels,pred)
    print("kendall:{:.3f}".format(k))
    print("preason:{:.3f}".format(p))
    print("spearman:{:.3f}".format(s))
    
def write_predictions_to_jsonl(predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for pred in predictions:
            jsonl_file.write(json.dumps({"pred_score": float(pred)}, ensure_ascii=False) + '\n')

def load_test_file(test_file_path):
    with open(test_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

# Load the predictions file
def load_predictions_file(predictions_file_path):
    with open(predictions_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

def main():
    # Define paths and parameters
    checkpoint_dir = 'run_4_lora_alpha_8_batch_size_16_lr_0.001/best_model/merged_model'  # Change to your checkpoint directory
    test_file = 'data/new_test_jsonl.jsonl'  # Path to your test file
    model_name = 'google/metricx-23-large-v2p0'
    max_input_length = 1024
    batch_size = 64
    is_qe=False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # Load model and tokenizer
    #model, tokenizer = load_checkpoint(checkpoint_dir, model_name)
    
    #prepare_dataset = Prepare_Dataset(test_file, tokenizer, max_input_length, device, is_qe)
    test_batch_size = 32  # Update with your batch size
    num_workers = 4  # Update with the number of workers
    #test_dataloader = prepare_dataset.create_dataloader(test_batch_size)

    # # Evaluate the model
    #preds=evaluate_model(model, tokenizer, test_dataloader, device)
    #print(len(preds))
    #print(preds)
    #write_predictions_to_jsonl(preds,'data/val_data.jsonl')
    df1 = load_test_file(test_file)
    df2 = load_predictions_file('data/vanilla_qe_test_r2_a4_lr_1e-3.jsonl')
    
    
    
    # print("Assamese:",compute_metrics(df2['pred_score'][:251],df1['score'][:251]))
    # print("Maithili",compute_metrics(df2['pred_score'][251:501],df1['score'][251:501]))
    # print("Kannada:",compute_metrics(df2['pred_score'][501:751],df1['score'][501:751]))
    # print("Punjabi:",compute_metrics(df2['pred_score'][751:],df1['score'][751:]))
    #print("Punjabi:",compute_metrics(df2['prediction'][:],df1['mqm'][:]))
    
    
    print("Hindi:",compute_metrics(df2['prediction'][:277],df1['score'][:277]))
    print("Malyalam",compute_metrics(df2['prediction'][277:553],df1['score'][277:553]))
    print(":",compute_metrics(df2['prediction'][553:829],df1['score'][553:829]))
    # 
    print("Tamil",compute_metrics(df2['prediction'][829:1105],df1['score'][829:1105]))
    print("Marathi",compute_metrics(df2['prediction'][1105:],df1['score'][1105:]))
    
if __name__ == "__main__":
    main()
