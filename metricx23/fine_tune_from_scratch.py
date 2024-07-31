"""fine-tuning MetricX model."""
'''
1. load dataset
2. load tokenizer, model
3. compute metrics function

'''
# Run fine_tune.sh to fine-tune model then merge_adapter.py will merge adapter weights with base model
# For fine- tuning and saving model ckpts files needed: fine_tine_from_scratch.py, model.py(prepare by metricx), prepare_dataset.py and merge_adapter.py
# For evaluating use predict.py it'll give some metricx style score(0: means perfect translation, 25: means worst translation)
# Need to write a script to convert out entire dataset in the form such that it has seg_id, system_name, label and predictions.
# That you can pass to evaluate.py and evaluate_wmt23.py 

##export PYTHONPATH=/data/anushka/metricx:$PYTHONPATH need to set python path 
# export HF_HOME=/data-3/anushka/.cache
# export HF_CACHE=/data-3/anushka/.cache

import dataclasses
import argparse
import json
from tqdm import tqdm
import os
from accelerate import Accelerator
import datasets
from metricx23 import models
from metricx23 import prepare_dataset
import torch
import transformers
import wandb
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

from scipy.stats import kendalltau, pearsonr, spearmanr
from peft import prepare_model_for_kbit_training
print("#####################################################")
print("fine tune from scratch")


def compute_metrics(pred,labels):
    print("############pred############", pred)
    #labels=pred.labels
    print("###################labels####",labels)
    #preds=pred.predictions
    # print("preds", preds)
    k, _ = kendalltau(labels,pred)
    p, _ = pearsonr(labels,pred)
    s, _ = spearmanr(labels,pred) #{'kendallTau': k}
    return k,p,s
        
def prepare_data(device,tokenizer,train_file,val_file,max_input_length,qe,batch_size):
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, use_fast= False)
    #print("########################## QE", qe)
    data_obj = prepare_dataset.Prepare_Dataset(
        train_file,
        val_file,
        tokenizer,
        max_input_length,
        device,
        qe
    )
    train_dataloader,val_dataloader= data_obj.create_dataloader(tokenizer=tokenizer, train_batch_size=batch_size, eval_batch_size=batch_size, num_workers=0)
    return train_dataloader,val_dataloader
          
def create_accelerator(gradient_accumulation_steps=1, log_with="wandb"):
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, log_with=log_with)
    if gradient_accumulation_steps > 1:
        accelerator.print("Gradient accumulation steps is", gradient_accumulation_steps)
    return accelerator


def load_model_and_tokenizer(model,tokenizer,accelerator):
    #device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device=accelerator.device
    #device = torch.device(device_name)
    accelerator.print(device)
    
    bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer , use_fast= False)
    model = models.MT5ForRegression.from_pretrained(model,device_map=device)
    accelerator.print(device)
    accelerator.print(model.config)
    return model,tokenizer

def lora_model(model,accelerator,r=4,a=64,target_modules=["q", "v"], lora_dropout=0.05, bias="none"):
    config = LoraConfig(r=r, lora_alpha=a, target_modules=target_modules, lora_dropout=lora_dropout, bias=bias)

    model = get_peft_model(model, config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()
    return model
def get_optimizer_scheduler(model, lr, epochs, dataloader_train, accelerator, gradient_accumulation_steps=4, use_8bit_adam=False):
    if use_8bit_adam:
        accelerator.print("Using 8 bit adam")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    else:
        optimizer = AdamW(model.parameters(), lr=lr,weight_decay=0.05)

    num_steps = int(len(dataloader_train)/accelerator.num_processes/gradient_accumulation_steps)*epochs
    warmup_steps = int(num_steps*0.05)
    print("gradient accumulation",gradient_accumulation_steps)

    accelerator.print("Total number of training steps is:", num_steps, "and warmup steps is:", warmup_steps)

    scheduler =  get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps*accelerator.num_processes, num_training_steps=num_steps*accelerator.num_processes, num_cycles=0.5)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps*accelerator.num_processes, num_training_steps=num_steps*accelerator.num_processes)

    return optimizer, scheduler, num_steps, warmup_steps

def prepare_model_optimizer_scheduler_dataloaders(model, optimizer, dataloader_train, dataloader_dev, scheduler, accelerator):
    model, optimizer, dataloader_train, dataloader_dev, scheduler = accelerator.prepare(model, optimizer, dataloader_train, dataloader_dev, scheduler)
    return model, optimizer, dataloader_train, dataloader_dev, scheduler

def train(model,tokenizer,dataloader_train, dataloader_dev, optimizer, scheduler, num_steps, epochs, accelerator, eval_every_n_steps=50, max_no_improvement_evals=100, save_path=None, is_deepspeed=False):
    time_to_stop=False
    max_no_improvement_evals=10
    no_improvement_evals = 0
    previous_eval_score = -float('inf')
    curr_step=0
    with tqdm(total=num_steps) as progress:
        for epoch in range(epochs):
            if accelerator.is_main_process:
                print("Epoch", epoch, "out of", epochs,"epochs")
            inner_step = 0
            for batch in dataloader_train:
                if curr_step % eval_every_n_steps == 0 and inner_step == 0:
                    accelerator.print("Evaluation step")
                    with torch.no_grad():
                        model.eval()
                        final_loss = 0
                        all_predictions = []
                        all_labels = []
    
                        num_eval_batches = len(dataloader_dev)
                        for eval_batch in tqdm(dataloader_dev):
                            eval_outputs = model(**eval_batch)
                            eval_loss = eval_outputs.loss
                            eval_loss_gathered = accelerator.gather(eval_loss)
                            final_loss += torch.mean(eval_loss_gathered)
                            
                            predictions = eval_outputs.predictions.cpu().numpy()
                            labels = eval_batch["labels"].cpu().numpy()
                            all_predictions.extend(predictions)
                            all_labels.extend(labels)
                            torch.cuda.empty_cache()
                            
                        all_predictions = torch.tensor(all_predictions).to(accelerator.device)
                        all_labels = torch.tensor(all_labels).to(accelerator.device)

                        gathered_predictions = accelerator.gather(all_predictions)
                        gathered_labels = accelerator.gather(all_labels)

                        final_loss = final_loss/num_eval_batches
                        final_loss = final_loss.item()
                        k,p,s = compute_metrics(gathered_predictions.cpu().numpy(), gathered_labels.cpu().numpy())
             
                        
                        if accelerator.is_main_process:
                            print("Eval loss after", curr_step, "steps is", final_loss)
                            wandb.log({"eval_loss": final_loss}, step=curr_step)
                            wandb.log({"kendalltau_score": k,"Pearson_score":p,"Spearman_score":s})

                        if p > previous_eval_score:
                            previous_eval_score = p
                            no_improvement_evals = 0
                            accelerator.print("Saving model since validation KendallTau reduced")
                            unwrapped_model = accelerator.unwrap_model(model)
                            state_dict = accelerator.get_state_dict(model)

                            # Explicitly save the model configuration
                            
                            unwrapped_model = accelerator.unwrap_model(model)
                            if is_deepspeed:
                                run_dir = os.path.join(save_path, wandb.run.name)
                                os.makedirs(run_dir, exist_ok=True)
                                unwrapped_model.config.save_pretrained(run_dir)
                                print(f"Best model saved to {run_dir}")
                                
                                unwrapped_model.save_pretrained(run_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model),)
                            else:
                                if accelerator.is_main_process:
                                    run_dir = os.path.join(save_path, wandb.run.name)
                                    os.makedirs(run_dir, exist_ok=True)
                                    unwrapped_model.config.save_pretrained(run_dir)
                                    print(f"Best model saved to {run_dir}")
                                    
                                    unwrapped_model.save_pretrained(run_dir,save_function=accelerator.save, state_dict=state_dict)
                                    tokenizer.save_pretrained(run_dir)
                                    accelerator.save_state(run_dir)
                        else:
                            no_improvement_evals += 1
                            accelerator.print("No improvement in eval kendallTau score for ", no_improvement_evals, " evaluations")
                            if no_improvement_evals >= max_no_improvement_evals:
                                accelerator.print("No improvement in eval kendallTau score for ", no_improvement_evals, " evaluations and max no improvement evals is ", max_no_improvement_evals, " so stopping")
                                time_to_stop = True
                    if time_to_stop:
                        break
                    model.train()
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    ## Gradient clipping
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    inner_step += 1
                if accelerator.sync_gradients:
                    inner_step = 0
                    if curr_step % 10 == 0:
                        accelerator.print("Loss at step", curr_step, "is", loss.detach().item())
                        if accelerator.is_main_process:
                            wandb.log({"loss": loss.detach().item(), "lr": scheduler.get_last_lr()[0]}, step=curr_step)
                    progress.update()
                    torch.cuda.empty_cache()
                    curr_step += 1
            if time_to_stop:
                break
        
        if not time_to_stop: ## Eval on the final checkpoint to see if its any better or not.
            accelerator.print("Model has not converged till the last batch so doing a final evaluation.")
            with torch.no_grad():
                model.eval()
                final_loss = 0
                all_predictions = []
                all_labels = []
    
                num_eval_batches = len(dataloader_dev)
                for eval_batch in tqdm(dataloader_dev):
                    eval_outputs = model(**eval_batch)
                    eval_loss = eval_outputs.loss
                    eval_loss_gathered = accelerator.gather(eval_loss)
                    final_loss += torch.mean(eval_loss_gathered)
                    
                    predictions = eval_outputs.predictions.cpu().numpy()
                    labels = eval_batch["labels"].cpu().numpy()
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)
                    
                all_predictions = torch.tensor(all_predictions).to(accelerator.device)
                all_labels = torch.tensor(all_labels).to(accelerator.device)

                gathered_predictions = accelerator.gather(all_predictions)
                gathered_labels = accelerator.gather(all_labels)

                final_loss = final_loss/num_eval_batches
                final_loss = final_loss.item()
                k,p,s = compute_metrics(gathered_predictions.cpu().numpy(), gathered_labels.cpu().numpy())
                
                if accelerator.is_main_process:
                            
                    print("Eval loss after", curr_step, "steps is", final_loss)
                    wandb.log({"eval_loss": final_loss}, step=curr_step)
                    
                    wandb.log({"kendalltau_score": k,"Pearson_score":p,"Spearman_score":s})

                if p > previous_eval_score:
                    previous_eval_score = p
                    no_improvement_evals = 0
                    accelerator.print("Saving model since validation KendallTau reduced")
                    unwrapped_model = accelerator.unwrap_model(model)
                    state_dict = accelerator.get_state_dict(model)

                    # Explicitly save the model configuration
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    if is_deepspeed:
                        run_dir = os.path.join(save_path, wandb.run.name)
                        os.makedirs(run_dir, exist_ok=True)
                        unwrapped_model.config.save_pretrained(run_dir)
                        print(f"Best model saved to {run_dir}")
                    
                    
                        unwrapped_model.save_pretrained(run_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model),)
                    else:
                        if accelerator.is_main_process:
                            run_dir = os.path.join(save_path, wandb.run.name)
                            os.makedirs(run_dir, exist_ok=True)
                            unwrapped_model.config.save_pretrained(run_dir)
                            print(f"Best model saved to {run_dir}")
                    
                    
                            unwrapped_model.save_pretrained(run_dir,save_function=accelerator.save, state_dict=state_dict)
                            tokenizer.save_pretrained(run_dir)
                            accelerator.save_state(run_dir)
                else:
                    accelerator.print("No improvement in eval loss even after the final batch.")
            
            
    



    
def main() -> None:
    sweep_config = {
    "name": "metricx-finetuning",
    "method": "bayes",
    'metric': {
        'name': 'kendalltau_score',
        'goal': 'maximize'
    },
    "parameters": {
        "epochs": {"values": [5]},
        "batch_size": {"values": [8,16]},
        "learning_rate": {"values": [0.001,0.003,0.00001]},
        "dropout": {"values": [0, 0.2, 0.3,0.05]},
        "gradient_accumulation_steps": {"values": [1]},
        "lora_r": {"values": [2, 4, 8]},
        "lora_alpha": {"values": [2, 4, 8]}
    }
}
    #sweep_id = wandb.sweep(sweep_config, project='metricx-finetuning')
    

    # Define the experiment configuration
    # experiment_config = {
    #     "model": args.model,
    #     "train_file": args.train_file,
    #     "val_file": args.val_file,
    #     "output_file_path": args.output_file_path,
    #     "tokenizer": args.tokenizer,
    #     "max_input_length": args.max_input_length,
    #     "epochs": args.epochs,
    #     "batch_size": args.batch_size,
    #     "gradient_accumulation_steps": args.gradient_accumulation_steps,
    #     "log_with": args.log_with,
    #     "num_workers": args.num_workers,
    #     "warmup_steps": args.warmup_steps,
    #     "learning_rate": args.lr,
    #     "lora": args.lora,
    #     "lora_r": args.lora_r,
    #     "lora_alpha": args.lora_alpha,
    #     "lora_dropout": args.lora_dropout,
    #     "bias": args.bias,
    #     "target_modules": args.target_modules,
    #     "is_deepspeed": args.is_deepspeed,
    #     "use_8bit_adam": args.use_8bit_adam,
    #     "qe":args.qe
    # }
    # Initialize wandb with the experiment configuration
    
    #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$",c)

    model = args.model
    train_file =args.train_file
    val_file =args.val_file
    output_file_path= args.output_file
    tokenizer= args.tokenizer
    max_input_length=args.max_input_length
    epochs= args.epochs
    batch_size=args.batch_size
    gradient_accumulation_steps=args.gradient_accumulation_steps
    log_with= args.log_with
    num_workers= args.num_workers
    warmup_steps=args.warmup_steps
    learning_rate=args.lr
    lora= args.lora
    lora_r=args.lora_r
    lora_alpha=args.lora_alpha
    lora_dropout=args.lora_dropout
    bias=args.bias
    target_modules= args.target_modules
    is_deepspeed= args.is_deepspeed
    use_8bit_adam=args.use_8bit_adam
    qe=args.qe


    # Verify if 'lora_r' and 'lora_alpha' are correctly set
    #if not hasattr(c, 'lora_r') or not hasattr(c, 'lora_alpha'):
    #    raise AttributeError("Config object does not have 'lora_r' or 'lora_alpha' attributes")

    # Set the run name
    
    # Prepare dataset and model
    accelerator = create_accelerator(gradient_accumulation_steps,log_with)
    device = accelerator.device
    if accelerator.is_main_process:
        wandb.init(project='fine-tuning_MetricX')
        if wandb.run is not None:
            wandb.run.name = f"run_{lora_r}_lora_alpha_{lora_alpha}_batch_size_{batch_size}_lr_{learning_rate}_qe_{qe}_transliterated"

    
    #device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #device = torch.device(device_name)
    accelerator.print(device)
    
    accelerator.print("Loading datasets")
    train_dataloader, val_dataloader = prepare_data(device,tokenizer,train_file,val_file,max_input_length,qe,batch_size)
    accelerator.print("Loading model")
    model, tokenizer = load_model_and_tokenizer(model,tokenizer, accelerator)

    if lora:
        accelerator.print("Converting to LoRA")
        model = lora_model(model, accelerator, lora_r, lora_alpha, target_modules, lora_dropout, bias)

    accelerator.print("Getting optimizer and scheduler")
    optimizer, scheduler, num_steps, warmup_steps = get_optimizer_scheduler(
        model, learning_rate, epochs, train_dataloader, accelerator, gradient_accumulation_steps, use_8bit_adam
    )

    accelerator.print("Preparing model, optimizer, scheduler, and dataloaders")
    model, optimizer, train_dataloader, val_dataloader, scheduler = prepare_model_optimizer_scheduler_dataloaders(
        model, optimizer, train_dataloader, val_dataloader, scheduler, accelerator
    )

    accelerator.print("Training and grad accumulation",gradient_accumulation_steps)
    #wandb.agent(sweep_id, function=train(model,tokenizer,train_dataloader, val_dataloader, optimizer, scheduler, num_steps, epochs, accelerator,50,5,output_file_path, is_deepspeed), count=5)
    #train(model,train_dataloader, val_dataloader, optimizer, scheduler, num_steps, c.epochs, accelerator,50,5,c.output_file_path, c.is_deepspeed)
    train(model,tokenizer,train_dataloader, val_dataloader, optimizer, scheduler, num_steps, epochs, accelerator,50,5,output_file_path, is_deepspeed)
    if accelerator.is_main_process:
        wandb.finish()



parser = argparse.ArgumentParser()
parser.add_argument('--wandb_project', type=str, default='metricx-finetuning')
parser.add_argument('--wandb_entity', type=str, default='cs22s015')
parser.add_argument('--train_file', type=str, default='data/train_data.jsonl')
parser.add_argument('--val_file', type=str, default='data/val_data.jsonl')
parser.add_argument('--output_file_path', type=str, default='ckpts/')
parser.add_argument('--model', type=str, default='google/metricx-23-xl-v2p0 ')
parser.add_argument('--tokenizer', type=str, default='google/mt5-xl')
parser.add_argument('--max_input_length', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--qe', type=int, default=1)

parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps", default=8)
parser.add_argument("--log_with", type=str, help="Log with", default='wandb')
parser.add_argument("--num_workers", type=int, help="Number of workers", default=32)
parser.add_argument("--warmup_steps", type=float, help="Warmup steps as a ratio of total steps", default=0.05)
parser.add_argument('-lr','--lr', type=float, default=0.0001)  

parser.add_argument("--lora", action="store_true", help="Use LoRA",default=True)
parser.add_argument("--lora_r", type=int, help="LoRA r", default=4)
parser.add_argument("--lora_alpha", type=int, help="LoRA alpha", default=8)
parser.add_argument("--lora_dropout", type=float, help="LoRA dropout", default=0.05)
parser.add_argument("--bias", type=str, help="Bias", default="none")
parser.add_argument("--target_modules", type=str, nargs="+", help="Target modules", default=["q", "v","k","o"])
parser.add_argument("--is_deepspeed", action="store_true", help="Is deepspeed")
parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8 bit adam")  

# parser.add_argument('-l','--loss', type=str, default='cross_entropy')
# parser.add_argument('-o','--optimizer', type=str, default='nadam')
# 
# parser.add_argument('-m','--momentum', type=float, default=0.9)
# parser.add_argument('-beta','--beta', type=float, default=0.95)
# parser.add_argument('-beta1','--beta1', type=float, default=0.9)
# parser.add_argument('-beta2','--beta2', type=float, default=0.999)
# parser.add_argument('-eps','--epsilon', type=float, default=1e-8)
# parser.add_argument('-w_d','--weight_decay', type=float, default=0)
# parser.add_argument('-w_i','--weight_init', type=str, default='he_uniform')
# parser.add_argument('-sz','--hidden_size', type=int, default=256)
# parser.add_argument('-nhl','--num_layers', type=int, default=5)
# parser.add_argument('-a','--activation', type=str, default='relu')
args = parser.parse_args()
    
if __name__ == "__main__":
    main()

