import dataclasses
import json
import os
import datasets
from metricx23 import models
import torch
import transformers


class Prepare_Dataset:
    def __init__(self, input_file: str, tokenizer, max_input_length: int, device, is_qe: bool, is_train:bool) -> None:
        self.input_file=input_file
        self.tokenizer=tokenizer
        self.max_input_length=max_input_length
        self.device=device
        self.is_qe=is_qe
        self.is_train=is_train
        
    def _make_input(self,example):
            if self.is_qe:
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

    def _tokenize(self,example):
        return self.tokenizer(
            example["input"],
            max_length=self.max_input_length,
            truncation=False,
            padding=True, 
        )

    def _remove_eos(self,example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    def _pad(self,example):
        #print("################### in")
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_input_length - len(input_ids))
        padded_attention_mask = attention_mask + [0] * (self.max_input_length - len(attention_mask))
        example["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
        example["attention_mask"] = torch.tensor(padded_attention_mask, dtype=torch.long)
        #print(len(example["input_ids"]))
        #print("################## out")
        return example    
    
    def get_dataset(self):
        """Gets the dataset for prediction.

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
        
        
        if self.is_train:
            ds = datasets.load_dataset("json", data_files={"train": self.input_file})
        #print(ds)
        #print("##################yes ds")
        else:
            ds = datasets.load_dataset("json", data_files={"val": self.input_file})
            #print(ds)
            #print("##################yes val")
                    
        ds = ds.map(self._make_input)
        ds = ds.map(self._tokenize)
        ds = ds.map(self._remove_eos)
        ds=ds.rename_column("mqm","labels")
        #ds = ds.map(_pad)
        #print("out ds")
        ds.set_format(
            type="torch",
            device=self.device)
        #print("set format ",ds)
        return ds

