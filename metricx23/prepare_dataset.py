import dataclasses
import json
import os
import datasets
from metricx23 import models
import torch
import transformers
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocessing

multiprocessing.set_start_method('spawn', force=True)


class Prepare_Dataset:
    def __init__(self, train_file: str,val_file:str, tokenizer, max_input_length: int, device, is_qe: bool) -> None:
        self.train_file=train_file
        self.tokenizer=tokenizer
        self.max_input_length=max_input_length
        self.device=device
        self.is_qe=is_qe
        self.val_file=val_file
               
    def _make_input(self,example):
        hypothesis = example.get("translation", "")
        if self.is_qe:
            #print("############################### make input qe")
            source = example.get("source", "")
            example["input"] = f"candidate: {hypothesis} source: {source}"
        else:
            reference = example.get("reference", "")
            example["input"] = f"candidate: {hypothesis} reference: {reference}"
        return example

    def _tokenize(self,example):
        
        return self.tokenizer(
            example["input"],
            max_length=self.max_input_length,
            padding=True, 
            truncation=True
        )

    def _remove_eos(self,example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example
 
    def map_dataset(self,ds):
        ds = ds.map(self._make_input)
        #print("make_input",ds['original'])
        ds = ds.map(self._tokenize)
        ds = ds.map(self._remove_eos)
        ds=ds.rename_column("mqm","labels")
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask","labels"],
            device=self.device)
        return ds
    def get_dataset(self):          
        tr = datasets.load_dataset("json", data_files={"train": self.train_file})['train']
        v = datasets.load_dataset("json", data_files={"val": self.val_file})['val']
        train=self.map_dataset(tr)
        val=self.map_dataset(v)
        # for i in range(5):
        #     example = val['val'][i]
        #     input_text = example['input']
        #     input_ids = example['input_ids']
        #     attention_mask = example['attention_mask']
            
        #     print(f"Example {i+1}:")
        #     print(f"Input: {input_text}")
        #     print(f"Input IDs: {input_ids}")
        #     print(f"Attention Mask: {attention_mask}")
        #     print()
        decoded_example = self.tokenizer.decode(val['input_ids'][0], skip_special_tokens=True)
        #print(f"Decoded Example 1: {decoded_example}")    
        #print("Val data",val)
        #print("train data",train)
            #print(ds)
            #print("##################yes val")
        return {"train":train,"val":val}
    def create_dataloader(self, tokenizer, train_batch_size, eval_batch_size, num_workers):
        dataset=self.get_dataset()
        data_collator =  transformers.DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader_train = DataLoader(dataset['train'], batch_size=train_batch_size, shuffle=True, collate_fn=data_collator, num_workers=num_workers)
        dataloader_dev = DataLoader(dataset['val'], batch_size=eval_batch_size, shuffle=False, collate_fn=data_collator, num_workers=num_workers)
        #print(dataloader_train,dataloader_dev)
        print("First batch example prepare data:", next(iter(dataloader_train)))
        return dataloader_train, dataloader_dev

   
   

