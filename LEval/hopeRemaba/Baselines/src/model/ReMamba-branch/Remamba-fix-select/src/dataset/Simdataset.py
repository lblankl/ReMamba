from transformers import LlamaTokenizerFast
from transformers import PretrainedConfig
import torch
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy
import sys
# sys.path.append('..')
from ..utils import first_mismatch
from .. import gist
class Preprocess():
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        
        


    def __call__(self,examples):
      
        
        text=examples['text']
        length=len(text)
        prompt=text[:length//2]
        response=text[length//2:]
        all_len=len(self.tokenizer.encode(text,add_special_tokens=False))+1

        examples["input"]=prompt
        examples["output"]=response
        examples["length"]=all_len
       
        
        return examples

class Postprocess():
    def __init__(self,tokenizer,mtokenizer,num_gist_tokens=1):
        self.tokenizer=tokenizer
        self.mamba_tokenizer=mtokenizer
        
        self.num_gist_tokens=num_gist_tokens
        self.gist_token=32000
        self.label_pad_token_id = -100
    def __call__(self,examples):
      
       
        
        
        model_inputs = {
            "input_ids": [],
            "labels": [],
            "attention_mask": [],
            "gist_ids": [],
            "prompt_input_ids": [],
            "prompt_attention_mask": [],
            "completion_input_ids": [],
            "completion_attention_mask": [],
            "prompt_input_idsm": []
        }
        
         

            

        instance=examples

        
        maybe_gist_str = " ".join(
            ["<GIST>" for _ in range(self.num_gist_tokens)]
        )
        
        
        prompt = f"{instance['input']}\n{maybe_gist_str}\n"  # noqa
        
        promptm= f"{instance['input']}"
        completion = f"{instance['output']}"
        
        tokenized_prompt = self.tokenizer(prompt)["input_ids"]
        tokenized_promptm = self.mamba_tokenizer(promptm)["input_ids"]

        tokenized_completion = self.tokenizer(completion, add_special_tokens=False)[
            "input_ids"
        ] + [self.tokenizer.eos_token_id]

        tokenized_completionm = self.mamba_tokenizer(completion, add_special_tokens=False)[
            "input_ids"
        ] + [self.mamba_tokenizer.eos_token_id]


        

        tokenized_source = tokenized_prompt + tokenized_completion
        tokenized_sourcem = tokenized_promptm + tokenized_completionm
        labels = [self.label_pad_token_id] * len(
            tokenized_promptm
        ) + tokenized_completionm
        
        
       
        model_inputs["input_ids"].append(tokenized_sourcem)
        model_inputs["labels"].append(labels)
        
        model_inputs["gist_ids"].append(tokenized_source)
        
        model_inputs["attention_mask"].append([1 for _ in tokenized_source])

        model_inputs["prompt_input_idsm"].append(tokenized_promptm)

        model_inputs["prompt_input_ids"].append(tokenized_prompt)
        model_inputs["prompt_attention_mask"].append([1 for _ in tokenized_prompt])

        model_inputs["completion_input_ids"].append(tokenized_completion)
        model_inputs["completion_attention_mask"].append(
            [1 for _ in tokenized_completion]
        )

        # Left-pad inputs, convert to tensor.
        for key, value in model_inputs.items():
            if key == "labels":
                pad_token_id = self.label_pad_token_id
            elif key == "input_ids" or key =="prompt_input_idsm":
                pad_token_id = self.mamba_tokenizer.pad_token_id
            else:
                pad_token_id = self.tokenizer.pad_token_id
            # To left-pad inputs, reverse, then right-pad, then reverse.
            value_tensors = [torch.tensor(v[::-1]) for v in value]
            
            model_inputs[key] = torch.fliplr(
                pad_sequence(
                    value_tensors,
                    batch_first=True,
                    padding_value=pad_token_id,
                )
            )
        # Construct gist mask.
    
        gist_fn = gist.make_gist_mask
       
        model_inputs["attention_mask_gist"] = gist_fn(
            model_inputs["gist_ids"],
            self.gist_token,
        )
        model_inputs["prompt_attention_mask_gist"] = gist_fn(
            model_inputs["prompt_input_ids"],
            self.gist_token,
        )
        
        for key, value in model_inputs.items():
            model_inputs[key]=value[0]
        
        return model_inputs

class CustomCollate:
    
    def __init__(self):
        pass
        #,padding="max_length",max_length=1024)

    def __call__(self, examples):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        out= {}
        #copy the keys from the first example
        for key in examples[0]:
            out[key] = []
        for example in examples:
            for key in example:
                out[key].append(example[key])
        #to torch tensor

        for key,value in out.items():
            
            out[key]=torch.tensor(value)
        
        return out