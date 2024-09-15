# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""
Gist training script, adapted from huggingface's run_clm.py example.
"""

import logging
import os


import torch  # noqa
from datasets import DatasetDict, load_dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    is_torch_tpu_available,
    set_seed,
)
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from model import GistLlama as gist_llama
from transformers import Seq2SeqTrainer,Trainer

from dataset.collator import DataCollatorForAlpacaCLM
from model.GistLlama import DEBUG_LLAMA_CONFIG, GistLlamaForCausalLM
from dataset.preprocessg import Preprocess as Preprocess
from trainer.CustomTrainer import CustomerTrainer


from peft import LoraConfig
from peft import get_peft_model

# Will error if the minimal version of Transformers is not installed. Remove at
# your own risks.
check_min_version("4.28.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)

class ModelArguments:
    def __init__(self):
      
        self.logging_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/log/llama7bgist"
        self.tokenizer= "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/meta/llama7b-chat"
        
        self.modelpath = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Model/General/llama-2-7b"
        self.output_dir= "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/llama7bgist"
        self.lora=16
        self.seed=42
        self.datapath="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/concept/conceptTransformer248/datasets/OpenOrca"
        self.cache_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/cache"
        self.mtype="llama"
        self.debug=False
        self.range=int(500e3)
def main():
    args=ModelArguments()
    training_args = TrainingArguments(
        deepspeed="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/deepspeedcfg/ds_config_zero3.json",
                                      output_dir=args.output_dir,
                                      num_train_epochs=1,
                                      overwrite_output_dir=True,per_device_train_batch_size=1,
                                      per_device_eval_batch_size=1,gradient_accumulation_steps=32,
                                      evaluation_strategy="steps",eval_steps=5,logging_steps=1,
                                      save_steps=60,save_total_limit=30,logging_dir=args.logging_dir,
                                      report_to="tensorboard",do_eval=True,
                                      bf16=True,
                                      dataloader_num_workers=2,remove_unused_columns=False) 
    set_seed(args.seed)

    
    train_datasets = load_dataset(
        args.datapath,
        cache_dir=args.cache_dir,
        split="train",
    ).select(range(0, args.range))
    eval_datasets = load_dataset(
            args.datapath,
            cache_dir=args.cache_dir,
            split="train",
        ).select(range(args.range, args.range+50))
    
    
    
   
    config = AutoConfig.from_pretrained(args.modelpath)
 

   
    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer
    )
    preprocess=Preprocess(tokenizer)
    train_datasets = train_datasets.map(preprocess,remove_columns=["system_prompt","question","response","id"])
    eval_datasets = eval_datasets.map(preprocess,remove_columns=["system_prompt","question","response","id"])
    train_datasets=train_datasets.filter(lambda x:(x["length"]<=config.max_position_embeddings))
    eval_datasets=eval_datasets.filter(lambda x:(x["length"]<=config.max_position_embeddings))
    if args.mtype=='llama':
        model_cls=GistLlamaForCausalLM
        vocab_size = gist_llama.PRETRAINED_VOCAB_SIZE
    elif args.mtype=='gemma':
        raise NotImplementedError("not implement")
    if args.debug==True:
        
        
        config.hidden_size=512
        config.intermediate_size=1024
        config.num_attention_heads=8
        config.num_hidden_layers=8
        config.num_key_value_heads=8
        
        
        model=model_cls(config)
    else:

        model = model_cls.from_pretrained(
            args.modelpath,
            cache_dir=args.cache_dir,
        )
    LoRAconfig = LoraConfig(
    target_modules=["q_proj",
    "k_proj","v_proj",
    ],
    modules_to_save=["embed_tokens","lm_head"],
    r=args.lora
    
    )
    
    

    # ==== BEGIN GIST CHANGES ====
    # Check if gist token has already been added to the model (e.g. because
    # we're resuming from a checkpoint.)
    if len(tokenizer) == vocab_size+1:
        pass
    else:
        # Initialize gist token
        tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
        model.resize_token_embeddings(len(tokenizer))
        # Set new word embedding to average of existing word embeddings. For why,
        # see https://nlp.stanford.edu/~johnhew/vocab-expansion.html
    
        with torch.no_grad():
                
            model.model.embed_tokens.weight[
                -1
            ] = model.model.embed_tokens.weight[:-1].mean(0)
            model.lm_head.weight[-1] = model.lm_head.weight[:-1].mean(0)
            
    gist_token = tokenizer.additional_special_tokens_ids[-1]
    tokenizer.pad_token_id = 0

    pmodel=get_peft_model(model,LoRAconfig)
    pmodel=pmodel.half()
    pmodel.print_trainable_parameters()
   
    # This data collator variant does causal language modeling with left
    # padding.
    data_collator = DataCollatorForAlpacaCLM(
        tokenizer,
        # Chosen so that <1% of examples are truncated.
        # See data/alpaca_plus/length_stats.txt for length stats.
        max_length=config.max_position_embeddings,  # source=256; target=256
        # Human eval examples are longer.
        max_length_human=config.max_position_embeddings,  # source=384; target=384
      
        num_gist_tokens=1,
        gist_token=gist_token,
        pad_token=tokenizer.pad_token_id,
        check_correctness=True,
    )
   
    
    
    

    trainer = Trainer(model=pmodel, args=training_args,train_dataset=train_datasets,
        eval_dataset=eval_datasets,data_collator=data_collator)
    trainer.train()
    trainer.save_model(output_dir=args.output_dir+'/end') 


if __name__ == "__main__":
    main()