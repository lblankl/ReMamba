
from transformers import PretrainedConfig,AutoTokenizer
import torch

IGNORE_INDEX = -100
class PretrainPreprocess():
    def __init__(self,tokenizer,response_token=None,max_len=6e3):
        self.tokenizer=tokenizer
        if response_token is None:
            self.response_token=[tokenizer.bos_token_id]
        else:
            self.response_token=response_token
        self.max_len=int(max_len)
    
        
        


    def __call__(self,examples):
        
        
        txt=examples['text']

        ipids=self.tokenizer.encode(txt,add_special_tokens=False)
        
        
        idlen=len(ipids)
        if idlen>self.max_len:
            idlen=self.max_len
            ipids=ipids[:idlen]
        seplen=idlen*3//4
        prompt = ipids[:seplen]

        response = ipids[seplen:]

        
        
        
        len_of_prompt=len(prompt)
        # len_of_response=len(response)
        
        
        
        sentence=prompt+response+[self.tokenizer.eos_token_id]
        
        exa={}
        
        exa["input_ids"]=sentence
        
        exa["labels"]=exa["input_ids"].copy()
        
        
        
        
        #set the prompt token to -100   
        exa["labels"][:len_of_prompt]=[IGNORE_INDEX]*len_of_prompt
        
        exa["len_of_prompt"]=len_of_prompt
        
        
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
        len_of_prompt=[e['len_of_prompt'] for e in examples]
        
        padded_input_ids=self.data_collator({"input_ids":input_ids})["input_ids"]
        attention_mask=self.data_collator({"input_ids":input_ids})["attention_mask"]

        padded_labels=self.data_collator({"input_ids":labels})["input_ids"]
        labels_attention_mask=self.data_collator({"input_ids":labels})["attention_mask"]
        
       

        labels_attention_mask=labels_attention_mask.to(torch.bool)
        
        padded_labels=padded_labels.masked_fill(~labels_attention_mask, IGNORE_INDEX)
        
        return {
            "input_ids":padded_input_ids,
            "attention_mask":attention_mask,
            "labels":padded_labels,
            "len_of_prompt":len_of_prompt
            
        }