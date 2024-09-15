
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
        # Dataset({
        #     features: ['input', 'file', 'instruction', 'output'],
        #     num_rows: 12000
        # })
        input=examples['instruction']
        output=examples['output']
        prompt=self.tokenizer.encode(input,add_special_tokens=True)
        response=self.tokenizer.encode(output,add_special_tokens=False)

        len_of_prompt=len(prompt)
        
        sentence=prompt+response+[self.tokenizer.eos_token_id]
        exa={}
        
        exa["input_ids"]=sentence
        
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
            "attention_mask":attention_mask,
            "labels":padded_labels
            
        }