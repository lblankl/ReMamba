




from accelerate import Accelerator

import os 

from mamba_ssm.utils.hf import load_config_hf
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from torch.utils.tensorboard import SummaryWriter

from transformers import (
    HfArgumentParser

)

from dataclasses import dataclass, field
from typing import Optional

import os


import torch  # noqa


from transformers import (
    AutoTokenizer
)



# from src.dataset.Dataset import Preprocess,CustomCollate

from src.dataset.NotemplateLong import CustomCollate




from src.model.config_remamba import ReMambaConfig
from src.model.ReMamba import ReMambaLMHeadModel
from datasets import concatenate_datasets
from datasets import load_from_disk
from peft import LoraConfig
from peft import get_peft_model

from transformers import get_scheduler


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None
    )
    mamba_path: Optional[str] = field(
        default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-hf'
    
    )
    mamba_peft: Optional[str] = field(
        default=None
      
    )
    remamba_peft: Optional[str] = field(
        default=None
    )
    
    
    remamba_config: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
 
    mamba_tokenizer: str = field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8btemplate", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    normal_data_range: Optional[int] = field(
        default=int(500e3), metadata={"help": "data range"}
    )
    longdataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use."}

    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use."}
    )
    dataset_name2: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use."}
    )
    dataset_name3: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use."}
    )
    lora : int = field(
        default=32, metadata={"help": "lora rank for remamba"}
    )
    
    cache_dir: Optional[str] = field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/cache", metadata={"help": "The cache directory for the datasets library"}
    )
    
    debugging: bool = field(
        default=False, metadata={"help": "debug mode"}
    )
    data_range: Optional[int] = field(
        default=None, metadata={"help": "data range"}
    )
    data_point: Optional[int] = field(
        default=None, metadata={"help": "data range"}
    )
    

@dataclass
class ConMambaTrainingArguments:
    name: Optional[str] = field(
        default="conmamba-llama7bgist-mamba2.8bf", metadata={"help": "The name of this project"}
    )
    logging_dir:  str= field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/log"
    )
    output_dir:  str= field(                                                                    
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/conmamba-llama7bgist-mamba2.8bf"
    )
    deepseed: Optional[str] = field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/deepspeedcfg/ds_config_zero3.json"
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "The batch size for training"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "The batch size for evaluation"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=32, metadata={"help": "The gradient accumulation steps"}
    )
    evaluation_strategy: Optional[str] = field(
        default="steps", metadata={"help": "The evaluation strategy"}
    )
    eval_steps: Optional[int] = field(
        default=int(500e3), metadata={"help": "The evaluation steps"}
    )#64
    logging_steps: Optional[int] = field(
        default=32, metadata={"help": "The logging steps"}
    )
    save_steps: Optional[int] = field(
        default=6000, metadata={"help": "The saving steps"}
    )
    save_total_limit: Optional[int] = field(
        default=30, metadata={"help": "The saving total limit"}
    )
    bf16: Optional[bool] = field(
        default=True, metadata={"help": "Whether use bf16"}
    )
    dataloader_num_workers: Optional[int] = field(
        default=2, metadata={"help": "The dataloader num workers"}
    )
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Whether remove unused columns"}
    )
    do_eval: Optional[bool] = field(
        default=True, metadata={"help": "Whether do evaluation"}
    )
    report_to: Optional[str] = field(
        default="tensorboard", metadata={"help": "The report to"}
    )
    overwrite_output_dir: Optional[bool] = field(
        default=True, metadata={"help": "Whether overwrite output directory"}
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "The number of training epochs"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "The seed"}
    )
    checkpoint: Optional[int] = field(
        default=None, metadata={"help": "The checkpoint"}
    )
    save_base: Optional[bool] = field(
        default=False, metadata={"help": "Whether do evaluation"}
    )
    grad_clip: Optional[float] = field(
        default=5.0, metadata={"help": "The checkpoint"}
    )
    select_rate_Max: Optional[float] = field(
        default=0.2, metadata={"help": "select rate for max"}

    )
    select_rate_Min: Optional[float] = field(
        default=0.1, metadata={"help": "select rate for min"}

    )
    ckpt: str = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConMambaTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    #cls
   

    #datasets
    train_datasets = load_from_disk(data_args.dataset_name).select(range(data_args.normal_data_range))
    long_datasets = load_from_disk(data_args.longdataset_name)
   
    train_datasets = concatenate_datasets([train_datasets, long_datasets])
    train_datasets=train_datasets.shuffle(42)
    data_len=train_datasets.shape[0]
    if data_args.data_point:
        rows,_=train_datasets.shape
        train_datasets = train_datasets.select(range(data_args.data_point,rows))
    if data_args.data_range is not None:
        data_len=data_args.data_range
        train_datasets = train_datasets.select(range(data_args.data_range))

   

    

    mamba_tokenizer=AutoTokenizer.from_pretrained(model_args.mamba_tokenizer)
   


    from torch.utils.data import DataLoader

    ##config
    
   
   
    if model_args.remamba_config:
        cfgdic =load_config_hf(model_args.remamba_config)
        cfg = ReMambaConfig(**cfgdic)
    else:
       
        cfg=None

    if data_args.debugging==True:
        
        
        
        

        cfg.d_model=512
        cfg.d_intermediate=1024
        
        cfg.n_layer=8
        # cfg.state_size=4
        # cfg.from_pretrain=True

    
        model=ReMambaLMHeadModel(config=cfg)
        
                    

        
        

    else:
        #model
        # model=AutoModelForCausalLM.from_pretrained(model_args.mamba_path)
        model=ReMambaLMHeadModel.from_pretrained(model_args.mamba_path,config=cfg)
        #init weight
        with torch.no_grad():
            model.backbone.K_proj[0].weight.data=model.backbone.layers[0].mixer.in_proj.weight.data[:cfg.d_model,:]
            model.backbone.K_proj[2].weight.data=model.backbone.layers[0].mixer.in_proj.weight.data[cfg.d_model:2*cfg.d_model,:]
            model.backbone.Q_proj[0].weight.data=model.backbone.layers[0].mixer.in_proj.weight.data[2*cfg.d_model:3*cfg.d_model,:]
            model.backbone.Q_proj[2].weight.data=model.backbone.layers[0].mixer.in_proj.weight.data[3*cfg.d_model:4*cfg.d_model,:]
            # model.backbone.G_proj.weight.data+=model.backbone.layers[0].mixer.in_proj.weight.data[3*cfg.d_model:4*cfg.d_model,:]
            
            
            model.backbone.V_proj[0].weight.data=model.backbone.layers[0].mixer.out_proj.weight.data[:,:cfg.d_model]
            model.backbone.V_proj[2].weight.data=model.backbone.layers[0].mixer.out_proj.weight.data[:,cfg.d_model:2*cfg.d_model]

        
            


   
    

  

    collator=CustomCollate(tokenizer=mamba_tokenizer)
    
    
    LoRAconfig = LoraConfig(
    target_modules=["embedding","out_proj","in_proj","lm_head"],
    
    modules_to_save=["Q_proj","V_proj","K_proj","ret_bias","dt_bias"],
    # modules_to_save=["embed_tokens","lm_head","expand","merge_proj"],
    r= data_args.lora,
    
    )
    output_dir=training_args.output_dir
    if training_args.save_base:
        model.save_pretrained(output_dir+"/"+"base")
    
    model=get_peft_model(model,LoRAconfig)
    # model.enable_input_require_grads()     # required for gradient checkpointing
    # model.gradient_checkpointing_enable()  # enable gradient checkpointing
    # model=model.half()
    model.print_trainable_parameters()
    # model.gradient_checkpointing_enable()
    
    
    epochs=training_args.num_train_epochs
    eval_steps=training_args.eval_steps
    batch_size=training_args.per_device_train_batch_size
    eval_batch_size=training_args.per_device_eval_batch_size
    
    logging_dir=training_args.logging_dir
    logging_steps=training_args.logging_steps
    save_steps=training_args.save_steps
    save_total_limit=2
    accumulation_step=training_args.gradient_accumulation_steps
    if training_args.checkpoint is not None:
        checkp=training_args.checkpoint
    else:
        checkp=0

    hps={"epochs":epochs,"batch_size":batch_size,"eval_batch_size":eval_batch_size,
         "output_dir":output_dir,"logging_dir":logging_dir,"eval_steps":eval_steps,
         "logging_steps":logging_steps,"save_steps":save_steps,"save_total_limit":save_total_limit}
    
    accelerator = Accelerator(log_with="tensorboard",project_dir=logging_dir,gradient_accumulation_steps=accumulation_step)
    
    accelerator.init_trackers(training_args.name,hps)

    dataloader=DataLoader(train_datasets,batch_size=batch_size,collate_fn=collator,num_workers=2)
    params=filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.1)
    seplr=2e-7
    
    #select parameters that contain Q_proj or V_proj or K_proj set lr=seplr
    #select other parameters set lr=2e-5

    # p=[]

    # sepparams=filter(lambda p: "Q_proj" in p[0] or "V_proj" in p[0] or "K_proj" in p[0], params)
    # for name,param in sepparams:
    #     p.append({"params":param,"lr":seplr})

    # otherparams=filter(lambda p: "Q_proj" not in p[0] and "V_proj" not in p[0] and "K_proj" not in p[0], params)
    # for name,param in otherparams:
    #     p.append({"params":param,"lr":2e-5})

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.1)
   
    #cosine learning rate scheduler initial lr=2e-5 weight decay=0.1
    
    num_training_steps=int(data_len//(accelerator.num_processes*batch_size*accumulation_step)*epochs)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,

    )
    
    # train_dataloader = dataloader
    
    train_dataloader, model,optimizer,lr_scheduler = accelerator.prepare(
        dataloader,model,optimizer,lr_scheduler
    )
    if training_args.ckpt:
        accelerator.load_state(training_args.ckpt)
        skipped_dataloader = accelerator.skip_first_batches(train_dataloader, checkp)
    if training_args.ckpt:
        dataloader=skipped_dataloader
    else:
        dataloader=train_dataloader
    steps=checkp
    log_steps=checkp//logging_steps
    
    state_num=0
    l_acu=0
  
    mode='normal'
    
    accelerator.wait_for_everyone()
   
    # accelerator.save_state(output_dir=training_args.output_dir+"/"+"state_start")
    # unwarp_model=accelerator.unwrap_model(model)
    # unwarp_model.backbone.set_forward_method(mode)
    for epoch in range(epochs):
        accelerator.print("epoch:",epoch)
        
        for batch in dataloader:
            
                
                steps+=1
                if steps>checkp:
                    model.train()
                    with accelerator.accumulate(model):
                        optimizer.zero_grad()
                        outputs = model(**batch)
                        loss=outputs.loss#+outputs.aux_loss
                       
                        selectloss=loss
                      
                        accelerator.backward(loss)
                        
                        accelerator.clip_grad_norm_(model.parameters(), training_args.grad_clip)
                        optimizer.step()
                        lr_scheduler.step()
                    
                    l_acu+=loss.item()/accumulation_step
                
                   
                    

                    if steps%logging_steps==0:
                      

                        if log_steps%100==0:
                            accelerator.print("training_loss",l_acu)
                            accelerator.print("select-loss",l_acu)
                        
                         
                        log_steps+=1
                        
                        
                        
                        
                        accelerator.log({"training_loss":l_acu},step=log_steps)
                        accelerator.log({"select_loss":l_acu},step=log_steps)
                       
                        

                    if steps%accumulation_step==0:
                        
        
                        l_acu=0
                
                     
                    # if steps%eval_steps==0:
                        
                    #     model.eval()
                    #     with torch.no_grad():
                    #         eval+=1
                    #         eval_loss=0
                    #         temp=0
                    #         for batch in test_dataloader:
                    #             # if batch["input_for_concept"].shape[1]<=1024:
                    #             outputs = model(**batch)
                    #             eloss=outputs.loss
                                
                    #             #perplexity=torch.exp(loss)
                                
                    #             all_loss=accelerator.gather_for_metrics(eloss)
                    #             all_loss=all_loss.mean()
                    #             eval_loss+=all_loss.item()
                    #             temp+=1
                    #         eval_loss=eval_loss/temp
                    #         accelerator.print("eval_loss:",eval_loss)
                    #         accelerator.log({"eval_loss":eval_loss},step=eval)
                    if steps%save_steps==0:
                        state_num+=1
                        # if state_num>save_total_limit:
                        #     #delete the oldest checkpoint floder
                        #     os.system("rm -rf "+output_dir+"/"+str(steps-save_total_limit*save_steps))
                        accelerator.wait_for_everyone()
                        unwarp_model=accelerator.unwrap_model(model)
                        # if args.peft==True :
                            
                        unwarp_model.save_pretrained(output_dir+"/"+str(steps),is_main_process=accelerator.is_main_process, save_function=accelerator.save
                        ,state_dict=accelerator.get_state_dict(model))
                    if steps%(save_steps*1)==0:
                        
                        accelerator.save_state(output_dir=training_args.output_dir+"/"+"state"+str(steps))

                    
    
    accelerator.wait_for_everyone()
    
    unwarp_model=accelerator.unwrap_model(model)
    unwarp_model.save_pretrained(output_dir+"/"+str(steps)+"end",is_main_process=accelerator.is_main_process, save_function=accelerator.save
    ,state_dict=accelerator.get_state_dict(model))

    accelerator.end_training()
    accelerator.print("end")

if __name__ == "__main__":
    
    
    main()