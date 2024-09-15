
from transformers import PretrainedConfig,AutoTokenizer
import torch

IGNORE_INDEX = -100
class Preprocess():
    def __init__(self,tokenizer,response_token=None):
        self.tokenizer=tokenizer
        if response_token is None:
            self.response_token=[tokenizer.bos_token_id]
        else:
            self.response_token=response_token
    
        
        


    def __call__(self,examples):
        
        systemprompt=examples["system_prompt"]
        question=examples["question"]
        response=examples["response"]

        
        messages = [
        {
            "role": "system",
            "content": systemprompt,
        },
        {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        

       
        len_of_prompt=len(prompt)
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sentence=prompt+response
        
        exa=self.tokenizer(sentence,add_special_tokens=False)
        exa["input_ids"]=exa["input_ids"]+[self.tokenizer.eos_token_id]
        
        exa["labels"]=exa["input_ids"].copy()
        
        
        
        
        #set the prompt token to -100   
        exa["labels"][:len_of_prompt]=[IGNORE_INDEX]*len_of_prompt
        
        
        
        
        return exa
    
from transformers import DataCollatorWithPadding


class CustomCollate:
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        #padding="max_length",max_length=1024)

    def __call__(self, examples):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        
        input_ids=[e["input_ids"] for e in examples]
        labels=[e["labels"] for e in examples]
        
        
        padded_input_ids=self.data_collator({"input_ids":input_ids})["input_ids"]
        attention_mask=self.data_collator({"input_ids":input_ids})["attention_mask"]

        padded_labels=self.data_collator({"input_ids":labels})["input_ids"]
        labels_attention_mask=self.data_collator({"input_ids":labels})["attention_mask"]
        
       

        labels_attention_mask=labels_attention_mask.to(torch.bool)
        
        padded_labels=padded_labels.masked_fill(~labels_attention_mask, IGNORE_INDEX)
        
        return {
            "input_ids":padded_input_ids,
            "labels":padded_labels
            
        }