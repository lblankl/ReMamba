model13Bpath="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Model/General/llama-2-13b"
m13_chat="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Model/General/llama-2-13b-chat"
model7Bpath="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Model/General/llama-2-7b"
m13Bopen="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-aipnlp/yuandl/param/13bopen"
m1B="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/concept/conceptTransformer248/param/1btiny"
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from peft import LoraConfig
import argparse
from transformers import TrainingArguments,Trainer
from transformers import LlamaTokenizer
from transformers import LlamaConfig
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments,AutoConfig
from transformers import LlamaForCausalLM
from dataset.Usualdataset import CustomCollate,Preprocess
import torch
# import deepspeed
# deepspeed.ops.op_builder.CPUAdamBuilder().load()
# from trainer.ConceptTrainer import CTrainer
from trainer.testtrainer import TestTrainer 
from transformers import AdamW
from peft import LoraConfig
from peft import get_peft_model
IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss
from transformers import TrainingArguments,Trainer
from datasets import load_from_disk
from datasets import concatenate_datasets
class ModelArguments:
    def __init__(self):
      
        self.logging_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/Open/remamba/hopemamba2.8b/log/mamba-long"
        self.tokenizer= "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8btemplate"
        
        self.modelpath = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-hf"
        self.output_dir= "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/Open/remamba/hopemamba2.8b/out/mamba-long"
        self.lora=32
        self.dataset_name='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/NormalOrcaRemambag'
        self.longdataset_name='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/LongContextRemamba'
        self.normal_data_range=300000
        self.data_range=None
       
        # self.ckpt="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/mamba2.8blongctx-Ti/checkpoint-1020"
def main():

    args=ModelArguments()
    
    
    training_args = TrainingArguments(
        deepspeed="deepspeedcfg/ds_config_zero3.json",
                                      output_dir=args.output_dir,
                                      num_train_epochs=1,
                                      overwrite_output_dir=True,per_device_train_batch_size=1,
                                      per_device_eval_batch_size=1,gradient_accumulation_steps=32,
                                      evaluation_strategy="steps",eval_steps=5,logging_steps=1,
                                      save_steps=60,save_total_limit=30,logging_dir=args.logging_dir,
                                      report_to="tensorboard",do_eval=True,
                                    #   include_inputs_for_metrics= True,
                                    
                                    bf16=True,
                                       
                                      dataloader_num_workers=2,remove_unused_columns=False)    
    #datasets
    data_args=args
    train_datasets = load_from_disk(data_args.dataset_name).select(range(data_args.normal_data_range))
    
    long_datasets = load_from_disk(data_args.longdataset_name)
    train_datasets = concatenate_datasets([train_datasets, long_datasets])
    train_datasets=train_datasets.shuffle(42)
    if data_args.data_range is not None:
        train_datasets = train_datasets.select(range(data_args.data_range))
    dataset=train_datasets
    test_dataset = load_from_disk(data_args.dataset_name).select(range(100))

    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer)
    config=AutoConfig.from_pretrained(args.modelpath)
    tokenizer.pad_token_id=config.pad_token_id
    
    
    
    # dataset=dataset.map(filter)
    
    # test_dataset=test_dataset.map(filter)
    

    

    
  

    LoRAconfig = LoraConfig(
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    
    r=args.lora,
    
    )
    
    ######
    # cfg=LlamaConfig.from_pretrained(model13Bpath)

    # cfg.hidden_size=512
    # cfg.intermediate_size=2048
    # cfg.num_attention_heads=16
    # cfg.num_hidden_layers=16
    # cfg.num_key_value_heads=8

    
    # model=LlamaForCausalLM(cfg)
    ##########
    model=AutoModelForCausalLM.from_pretrained(args.modelpath)
    pmodel=get_peft_model(model,LoRAconfig)
    pmodel=pmodel.half()
    pmodel.print_trainable_parameters()
    
    def compute_metrics(pred):
        
        # logits, labels = pred 
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = labels[...,1:].contiguous()
        #         # Flatten the tokens
        # loss_fct = torch.nn.CrossEntropyLoss()
        # shift_logits = shift_logits.view(-1, vocab_size)
        # shift_labels = shift_labels.view(-1)
        # shift_labels = shift_labels.to(shift_logits.device)
        # loss = loss_fct(shift_logits, shift_labels)
        loss=0/0
        return {"eval_loss":loss}
    
    trainer = TestTrainer(model=pmodel, args=training_args,train_dataset=dataset,
                       eval_dataset= test_dataset,data_collator=CustomCollate(tokenizer=tokenizer))
    trainer.train()
    trainer.save_model(output_dir=args.output_dir+'/end')
    
if __name__ == "__main__":
    
    main()
#NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"