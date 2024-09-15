
from transformers import PretrainedConfig,LlamaTokenizer
import torch
model13Bpath="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Model/General/llama-2-13b"
IGNORE_INDEX = -100
class Filter():
    def __init__(self,tokenizer=LlamaTokenizer.from_pretrained(model13Bpath),response_token=None):
        self.tokenizer=tokenizer
        if response_token is None:
            self.response_token=[tokenizer.bos_token_id]
        else:
            self.response_token=response_token
    
        
        


    def __call__(self,examples):
        
        systemprompt=examples["system_prompt"]
        question=examples["question"]
        response=examples["response"]

        
        p=self.tokenizer(systemprompt+question)["input_ids"]
       
        len_of_prompt=len(p)
        sentence=systemprompt+question+response
        
        exa=self.tokenizer(sentence,add_special_tokens=False)
        exa["input_ids"]=exa["input_ids"]+[self.tokenizer.eos_token_id]
        
        
        examples["input_ids"]=exa["input_ids"]
        
        
        return examples


