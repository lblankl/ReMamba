from transformers import (
    HfArgumentParser,
    Trainer,
)

from dataclasses import dataclass, field
from typing import Optional

from transformers import LlamaConfig,MambaConfig,GemmaConfig,PretrainedConfig
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

from src.model import GistLlama as gistllama
from src.model import GistGemma as gistgemma
from transformers import Seq2SeqTrainer,Trainer

from src.dataset.collator import DataCollatorForConMamba,DataCollatorForAlpacaCLM
from src.model.GistLlama import DEBUG_LLAMA_CONFIG, GistLlamaForCausalLM
from src.dataset.preprocess import Preprocess as Preprocess
from src.trainer.CustomTrainer import CustomerTrainer

from transformers.models.mamba.modeling_mamba import MambaForCausalLM

from src.model.configuration_ConMamba import ConMambaConfig
from src.model.modeling_ConMamba import ConMambaForCausalLM,ConMambaModel

from src.model.GistGemma import GistGemma
from src.model.GistLlama import GistLlama
from peft import LoraConfig,PeftModel
from peft import get_peft_model
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Only useful in unified mode"
            )
        },
    )
    model_type: Optional[str] = field(
        default='seperate',
        metadata={"help": "whether initialize from seperated or unified"},
    )

    tmodel_type: Optional[str] = field(
        default='gistllama',
        metadata={"help": "transformers' model type"},
    )

    transformer_path: Optional[str] = field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Model/General/llama-2-7b",
        metadata={
            "help": (
                "model path for transformers"
            )
        },
    )
    transformer_peft: Optional[str] = field(
        default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/llama7bgist/end',
        metadata={
            "help": (
                "peft path for transformers"
            )
        },
    )
    mamba_path: Optional[str] = field(
        default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-hf',
        metadata={
            "help": (
                "model path for mamba"
            )
        },
    )
    mamba_peft: Optional[str] = field(
        default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/mamba2.8b/end',
        metadata={
            "help": (
                "peft path for mamba"
            )
        },
    )


    conmamba_config: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    transformer_tokenizer: str = field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/meta/llama7b-chat", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    mamba_tokenizer: str = field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8btemplate", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/concept/conceptTransformer248/datasets/OpenOrca", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    
    # logging_dir:  str= field(
    #     default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/log/conmamba-llama7bgist-mamba2.8b", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    # )
    # output_dir:  str= field(                                                                    
    #     default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/conmamba-llama7bgist-mamba2.8b", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    # )
    lora : int = field(
        default=16, metadata={"help": "lora rank for conmamba"}
    )
    # seed : int = field(
    #     default=42, metadata={"help": "seed"}
    # )
    #"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/cache"
    cache_dir: Optional[str] = field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/cache", metadata={"help": "The cache directory for the datasets library"}
    )
    debugging: bool = field(
        default=False, metadata={"help": "debug mode"}
    )
    data_range: Optional[int] = field(
        default=int(500e3), metadata={"help": "data range"}
    )
    num_gist_tokens: Optional[int] = field(
        default=1, metadata={"help": "data range"}
    )
"""
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
"""
#set above to default
@dataclass
class ConMambaTrainingArguments(TrainingArguments):
    logging_dir:  str= field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/log/conmamba-llama7bgist-mamba2.8b", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    output_dir:  str= field(                                                                    
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/conmamba-llama7bgist-mamba2.8b", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    deepseed: Optional[str] = field(
        default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/deepspeedcfg/ds_config_zero3.json", metadata={"help": "The deepseed config file for training"}
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
        default=5, metadata={"help": "The evaluation steps"}
    )
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "The logging steps"}
    )
    save_steps: Optional[int] = field(
        default=60, metadata={"help": "The saving steps"}
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
  
    # def __post_init__(self):
    #     # super.__post_init__()
    #     self.deepseed = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/deepspeedcfg/ds_config_zero3.json"
    #     self.per_device_train_batch_size = 1
    #     self.per_device_eval_batch_size = 1
    #     self.gradient_accumulation_steps = 32
    #     self.evaluation_strategy = "steps"
    #     self.eval_steps = 5
    #     self.logging_steps = 1
    #     self.save_steps = 60
    #     self.save_total_limit = 30
    #     self.bf16 = True
    #     self.dataloader_num_workers = 2
    #     self.remove_unused_columns = False
    #     self.do_eval = True
    #     self.report_to = "tensorboard"
    #     self.overwrite_output_dir = True
    #     self.num_train_epochs = 1
    #     self.seed=42
        # self.output_dir = self.output_dir
        # self.logging_dir = self.logging_dir
     




def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConMambaTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    COMABA_CLS = {
    "gistllama": GistLlama,
    "gistgemma": GistGemma,

    }
    VOC_CLS = {
    "gistllama": gistllama,
    "gistgemma": gistgemma,

    }

    #dataset
    train_datasets = load_dataset(
        data_args.dataset_name,
        cache_dir=data_args.cache_dir,
        split="train",
    ).select(range(data_args.data_range, 2*data_args.data_range))
    eval_datasets = load_dataset(
            data_args.dataset_name,
            cache_dir=data_args.cache_dir,
            split="train",
    ).select(range(data_args.data_range, data_args.data_range+50))
    #data_args.data_range+

    #tokenizer
    transformer_tokenizer=AutoTokenizer.from_pretrained(model_args.transformer_tokenizer)
    
    vocab_size = VOC_CLS[model_args.tmodel_type].PRETRAINED_VOCAB_SIZE

    if len(transformer_tokenizer) == vocab_size+1:
        pass
    else:
        # Initialize gist token
        
        transformer_tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})

    

    mamba_tokenizer=AutoTokenizer.from_pretrained(model_args.mamba_tokenizer)
    transformer_tokenizer.chat_template=mamba_tokenizer.chat_template
    

    ##config
    
   
    model_cls=COMABA_CLS[model_args.tmodel_type]
    if model_args.conmamba_config:
        cfg=ConMambaConfig.from_pretrained(model_args.conmamba_config)
    else:
        
        transformer_config=AutoConfig.from_pretrained(model_args.transformer_path)
        mamba_config=AutoConfig.from_pretrained(model_args.mamba_path)
        cfg=ConMambaConfig(_transformers_implementation=model_args.tmodel_type,mambacfg=mamba_config,transformerscfg=transformer_config)
    
    if model_args.model_name_or_path:
        model=ConMambaForCausalLM(config=cfg)
    else:
        if data_args.debugging==True:
            
            
            
            cfg.transformerscfg.hidden_size=512
            cfg.transformerscfg.intermediate_size=1024
            cfg.transformerscfg.num_attention_heads=8
            cfg.transformerscfg.num_hidden_layers=8
            cfg.transformerscfg.num_key_value_heads=8

            cfg.mambacfg.hidden_size=512
            cfg.mambacfg.intermediate_size=1024
            cfg.mambacfg.n_layer=8
            cfg.mambacfg.num_hidden_layers=8
            cfg.mambacfg.state_size=4
            cfg.from_pretrain=True

            # model=MambaForCausalLM(config=cfg.mambacfg)
            # model=MambaForCausalLM.from_pretrained(model_args.mamba_path)
            model=ConMambaForCausalLM(config=cfg)
            if  model.transformers.model.embed_tokens.weight.shape[0]== vocab_size + 1:
                pass
            else:       
                model.transformers.resize_token_embeddings(len(transformer_tokenizer))
                # Set new word embedding to average of existing word embeddings. For why,
                # see https://nlp.stanford.edu/~johnhew/vocab-expansion.html

                with torch.no_grad():
                        
                    model.transformers.model.embed_tokens.weight[
                        -1
                    ] = model.transformers.model.embed_tokens.weight[:-1].mean(0)
            

            
            

        else:
            #model
            model=ConMambaForCausalLM(config=cfg)
            model.mamba=ConMambaModel.from_pretrained(model_args.mamba_path,config=cfg)
            if model_args.mamba_peft:
                model.mamba=PeftModel.from_pretrained(model.mamba,model_args.mamba_peft)
                model.mamba.merge_and_unload()

                
            model.transformers=model_cls.from_pretrained(model_args.transformer_path)

            if  model.transformers.model.embed_tokens.weight.shape[0]== vocab_size + 1:
                pass
            else:
                model.transformers.resize_token_embeddings(len(transformer_tokenizer))

            if model_args.transformer_peft:

                model.transformers=PeftModel.from_pretrained(model.transformers,model_args.transformer_peft)
                model.transformers.merge_and_unload()
            
            
    
                
        
    gist_token = transformer_tokenizer.additional_special_tokens_ids[-1]
    model.gist_token=gist_token
    model.num_gist_tokens=data_args.num_gist_tokens
   
    #preprocess and filter
    transformer_tokenizer.pad_token_id=cfg.transformerscfg.pad_token_id
    preprocess=Preprocess(transformer_tokenizer)
    train_datasets = train_datasets.map(preprocess,remove_columns=["system_prompt","question","response","id"])
    eval_datasets = eval_datasets.map(preprocess,remove_columns=["system_prompt","question","response","id"])

    train_datasets=train_datasets.filter(lambda x:(x["length"]<=1024))
    eval_datasets=eval_datasets.filter(lambda x:(x["length"]<1024))

    collator=DataCollatorForConMamba(
            transformer_tokenizer,
            mamba_tokenizer,
            max_length=cfg.transformerscfg.max_position_embeddings,
            max_length_human=cfg.transformerscfg.max_position_embeddings,
            num_gist_tokens=data_args.num_gist_tokens,
            gist_token=gist_token,
            pad_token=0,
            check_correctness=False,
    )
    # collator = DataCollatorForAlpacaCLM(
    #     mamba_tokenizer,
    #     # Chosen so that <1% of examples are truncated.
    #     # See data/alpaca_plus/length_stats.txt for length stats.
    #     max_length=cfg.transformerscfg.max_position_embeddings,
    #     # Human eval examples are longer.
    #     max_length_human=cfg.transformerscfg.max_position_embeddings,
      
    #     num_gist_tokens=1,
    #     gist_token=gist_token,
    #     pad_token=0,
    #     check_correctness=False,
    # )

    LoRAconfig = LoraConfig(
    target_modules=["q_proj",
    "k_proj","v_proj","x_proj", "embeddings", "out_proj"
    ],
    modules_to_save=["embed_tokens","lm_head","in_proj","expand","merge_proj"],
    r= data_args.lora,
    
    )
    #,"in_proj"
    # pmodel=model
    pmodel=get_peft_model(model,LoRAconfig)
    pmodel=pmodel.half()
    pmodel.print_trainable_parameters()

    trainer = Trainer(model=pmodel, args=training_args,train_dataset=train_datasets,
        eval_dataset=eval_datasets,data_collator=collator)
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir+'/end') 


if __name__ == "__main__":
    main()