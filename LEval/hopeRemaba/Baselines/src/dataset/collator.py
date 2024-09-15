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

logger = logging.getLogger(__name__)
# class SegmentMethod:
#     def __init__(self,num_segments,max_position_embeddings):
#         self.

# def random_segment_lengths(self, input_ids, num_segments,max_position_embeddings):
#         """Returns a list of random segment lengths that sum up to num_segments"""
#         max_positions = max_position_embeddings
#         if num_segments > 1:
#             min_segment_length = max(math.ceil((input_ids.size(1) - max_positions) / (num_segments - 1)), 2)

#             total_variable_length = input_ids.size(1) - min_segment_length * num_segments
#             if num_segments - 1 > total_variable_length:
#                 raise ValueError(f"The specified number of segments_per_substep cannot cover the entire input sequence.")
#             breakpoints = torch.multinomial(torch.ones(total_variable_length), num_segments - 1)
#             segment_lengths = torch.diff(breakpoints.sort(-1).values,
#                                         prepend=torch.tensor([0]),
#                                         append=torch.tensor([total_variable_length]))
#             segment_lengths = (segment_lengths + min_segment_length).tolist()
#         else:
#             segment_lengths = [input_ids.size(1)]
#         return segment_lengths


@dataclass
class DataCollatorForAlpacaCLM:
    """Data collator for decoder-only models. Does left padding."""

    tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    max_length_human: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    gist_token: int = 32000
    pad_token: int = 0
    add_gist_token: bool = True
    gist_condition: str = "gist"
    num_gist_tokens: int = 1
    check_correctness: bool = False
    
    def __post_init__(self):
        if self.max_length_human is None:
            self.max_length_human = self.max_length

    def __call__(self, batch, return_tensors=None):
        

        

        max_length = self.max_length
        if return_tensors is None:
            return_tensors = self.return_tensors
        model_inputs = defaultdict(list)
        for instance in batch:
            if not self.add_gist_token:
                # Add gist tokens later, during tokenization.
                maybe_gist_str = ""
            else:
                maybe_gist_str = " ".join(
                    ["<GIST>" for _ in range(self.num_gist_tokens)]
                )
            
            
            # prompt = f"{instance['input']}\n{maybe_gist_str}\n"  # noqa
            
            # completion = f"{instance['output']}"

            # tokenized_prompt = self.tokenizer(prompt)["input_ids"]

            tokenized_source = [self.tokenizer.bos_token_id]+instance['input_ids']+ [self.tokenizer.eos_token_id]
            

            labels = tokenized_source
            input_ids=torch.tensor([tokenized_source[1:-1]])

            model_inputs["input_ids"].append(tokenized_source)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1 for _ in tokenized_source])

            # model_inputs["prompt_input_ids"].append(tokenized_prompt)
            # model_inputs["prompt_attention_mask"].append([1 for _ in tokenized_prompt])

            # model_inputs["completion_input_ids"].append(tokenized_completion)
            # model_inputs["completion_attention_mask"].append(
            #     [1 for _ in tokenized_completion]
            # )

        # Left-pad inputs, convert to tensor.
        for key, value in model_inputs.items():
            if key == "labels":
                pad_token_id = self.label_pad_token_id
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
        if self.gist_condition == "gist":
            gist_fn = gist.make_gist_mask
        elif self.gist_condition == "neg_control":
            gist_fn = gist.make_neg_control_mask
        elif self.gist_condition == "pos_control":
            gist_fn = gist.make_pos_control_mask
        else:
            raise ValueError(f"Unknown gist condition {self.gist_condition}")
        model_inputs["attention_mask_gist"] = gist_fn(
            model_inputs["input_ids"],
            self.gist_token,
        )
        model_inputs["prompt_attention_mask_gist"] = gist_fn(
            model_inputs["prompt_input_ids"],
            self.gist_token,
        )

        return model_inputs




@dataclass
class DataCollatorForConMamba:
    """Data collator for decoder-only models. Does left padding."""

    tokenizer: PreTrainedTokenizerBase
    mamba_tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    max_length_human: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    gist_token: int = 32000
    pad_token: int = 0
    add_gist_token: bool = True
    gist_condition: str = "gist"
    num_gist_tokens: int = 1
    check_correctness: bool = False
    
    def __post_init__(self):
        if self.max_length_human is None:
            self.max_length_human = self.max_length

    def __call__(self, batch, return_tensors=None):
        

        

        max_length = self.max_length
        if return_tensors is None:
            return_tensors = self.return_tensors
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
        
        for instance in batch:

            



            if not self.add_gist_token:
                # Add gist tokens later, during tokenization.
                maybe_gist_str = ""
            else:
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
        if self.gist_condition == "gist":
            gist_fn = gist.make_gist_mask
        elif self.gist_condition == "neg_control":
            gist_fn = gist.make_neg_control_mask
        elif self.gist_condition == "pos_control":
            gist_fn = gist.make_pos_control_mask
        else:
            raise ValueError(f"Unknown gist condition {self.gist_condition}")
        
        model_inputs["attention_mask_gist"] = gist_fn(
            model_inputs["gist_ids"],
            self.gist_token,
        )
        model_inputs["prompt_attention_mask_gist"] = gist_fn(
            model_inputs["prompt_input_ids"],
            self.gist_token,
        )
        

        return model_inputs




@dataclass
class DataCollatorForConMambaInference:
    """Data collator for decoder-only models. Does left padding."""

    tokenizer: PreTrainedTokenizerBase
    mamba_tokenizer: PreTrainedTokenizerBase
    max_length: Optional[int] = None
    max_length_human: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    gist_token: int = 32000
    pad_token: int = 0
    add_gist_token: bool = True
    gist_condition: str = "gist"
    num_gist_tokens: int = 1
    check_correctness: bool = False
    
    def __post_init__(self):
        if self.max_length_human is None:
            self.max_length_human = self.max_length

    def __call__(self, batch, return_tensors=None):
        

        

        max_length = self.max_length
        if return_tensors is None:
            return_tensors = self.return_tensors
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "gist_ids": [],
            "prompt_input_ids": [],
            "prompt_attention_mask": [],
            "prompt_input_idsm": []
        }
        
        for instance in batch:

            



            if not self.add_gist_token:
                # Add gist tokens later, during tokenization.
                maybe_gist_str = ""
            else:
                maybe_gist_str = " ".join(
                    ["<GIST>" for _ in range(self.num_gist_tokens)]
                )
            
            
            prompt = f"{instance['input']}\n{maybe_gist_str}\n"  # noqa
            
            promptm= f"{instance['input']}"

            # completion = f"{instance['output']}"
            
            tokenized_prompt = self.tokenizer(prompt,add_special_tokens=False)["input_ids"]
            tokenized_promptm = self.mamba_tokenizer(promptm,add_special_tokens=False)["input_ids"]

            # tokenized_completion = self.tokenizer(completion, add_special_tokens=False)[
            #     "input_ids"
            # ] + [self.tokenizer.eos_token_id]

            # tokenized_completionm = self.mamba_tokenizer(completion, add_special_tokens=False)[
            #     "input_ids"
            # ] + [self.mamba_tokenizer.eos_token_id]


         

            tokenized_source = tokenized_prompt 
            tokenized_sourcem = tokenized_promptm 
            
            
           

            model_inputs["input_ids"].append(tokenized_sourcem)
            
            
            model_inputs["gist_ids"].append(tokenized_source)
            
            model_inputs["attention_mask"].append([1 for _ in tokenized_source])

            model_inputs["prompt_input_idsm"].append(tokenized_promptm)

            model_inputs["prompt_input_ids"].append(tokenized_prompt)
            model_inputs["prompt_attention_mask"].append([1 for _ in tokenized_prompt])

            # model_inputs["completion_input_ids"].append(tokenized_completion)
            # model_inputs["completion_attention_mask"].append(
            #     [1 for _ in tokenized_completion]
            # )

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
        if self.gist_condition == "gist":
            gist_fn = gist.make_gist_mask
        elif self.gist_condition == "neg_control":
            gist_fn = gist.make_neg_control_mask
        elif self.gist_condition == "pos_control":
            gist_fn = gist.make_pos_control_mask
        else:
            raise ValueError(f"Unknown gist condition {self.gist_condition}")
        
        model_inputs["attention_mask_gist"] = gist_fn(
            model_inputs["gist_ids"],
            self.gist_token,
        )
        # model_inputs["prompt_attention_mask_gist"] = gist_fn(
        #     model_inputs["prompt_input_ids"],
        #     self.gist_token,
        # )
        del model_inputs["prompt_input_ids"]
        del model_inputs["prompt_attention_mask"]

        return model_inputs