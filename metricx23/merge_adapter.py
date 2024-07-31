import argparse
import json
from metricx23 import models
import torch
import transformers
from peft import PeftConfig, PeftModel
import bitsandbytes as bnb

from bitsandbytes.functional import dequantize_4bit
from peft.utils import _get_submodules
import copy


def dequantize_model(model, dtype=torch.bfloat16, device="cuda"):
    """
    'model': the peftmodel you loaded with qlora.
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """
    cls = bnb.nn.Linear4bit
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)

                quant_state.dtype = dtype

                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)
        # to save model, you have to unset this attribute
        model.is_loaded_in_4bit = False
        
        return model

def load_checkpoint(ckpt, model_name,output_dir):
    # Load model and tokenizer from checkpoint directory
    peft_config = PeftConfig.from_pretrained(ckpt)
    print("loading base model......")
    quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    base_model =  models.MT5ForRegression.from_pretrained(
            model_name ,
            torch_dtype=torch.bfloat16
        )
    base_model = dequantize_model(base_model)
    print("Loading the lora model...")
    lora_model = PeftModel.from_pretrained(base_model, ckpt)
    print("Merging the lora modules...")
    merged_model = lora_model.merge_and_unload()
    tokenizer = transformers.AutoTokenizer.from_pretrained('google/mt5-large' , use_fast= False)
    
    #print("Converting to bfloat16...")
    #merged_model = merged_model.to(dtype=torch.bfloat16)   
    print(f"Saving to {output_dir}...")
    merged_model.save_pretrained(output_dir) 
    tokenizer.save_pretrained(output_dir)

def main():
    load_checkpoint(args.checkpoint,args.base_model,args.output_dir)
parser = argparse.ArgumentParser()
parser.add_argument('-bm','--base_model', type=str, default='google/metricx-23-large-v2p0')
parser.add_argument('-ckpt','--checkpoint', type=str, default='ckpts/run_2_lora_alpha_4_batch_size_16_lr_0.001_qe_1')
parser.add_argument('-out','--output_dir', type=str, default='ckpts/run_2_lora_alpha_4_batch_size_16_lr_0.001_qe_1/merged_model')

args = parser.parse_args()
    
if __name__ == "__main__":
    main()
    
    
