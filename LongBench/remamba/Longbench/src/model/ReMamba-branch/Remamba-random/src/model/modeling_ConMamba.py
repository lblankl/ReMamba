"""PyTorch MAMBA model."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from typing import List
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ..dataset.collator import DataCollatorForConMambaInference
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available

from .configuration_ConMamba import ConMambaConfig


from ..gist_caching import GistActivations
from .GistGemma import GistGemma
from .GistLlama import GistLlama
# from .Mamba import MambaModel
from transformers.models.mamba.modeling_mamba import MambaModel,MambaPreTrainedModel

logger = logging.get_logger(__name__)

if is_mamba_ssm_available():
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)

_CHECKPOINT_FOR_DOC = "state-spaces/mamba-130m-hf"
_CONFIG_FOR_DOC = "MambaConfig"



# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os

from collections import namedtuple

import torch
import torch.nn as nn

from .mambanohf.config_mamba import MambaConfig
from .mambanohf.mamba_simple import Mamba, Block
from .generationNew import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
class MambaCache:
    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
@dataclass
class MambaCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

class HiddenMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        
        

class LastHiddenMerger(HiddenMerger):
    def __init__(self, config):
        super().__init__(config)
        self.config=config
        
        self.expand=nn.Linear(config.transformerscfg.hidden_size,config.mambacfg.d_model)
        self.merge_proj=nn.Linear(config.mambacfg.d_model*2,config.mambacfg.d_model)
        # self.act=ACT2FN[config.transformerscfg.hidden_act]
    def forward(self,gist_activations,hidden_states,prompt_input_idsm,idx):
        prompt_index=prompt_input_idsm.shape[-1]-1
        gist_indices=gist_activations.gist_indices
        gist_hiddens=gist_activations.all_last_hidden_state[:,idx,0:1,:]
        
        for i in range(hidden_states.shape[0]):
            
            hidden_state=hidden_states[i][prompt_index:prompt_index+1,:]
            
            expanded_hidden=self.expand(gist_hiddens[i])
           
            dtype= expanded_hidden.dtype
           
            concat=torch.cat((hidden_state,expanded_hidden),1).to(dtype)
            # concat=self.act(concat)
            hidden_states[i][prompt_index,:]=self.merge_proj(concat).to(hidden_states.dtype)

        return hidden_states

class EffT(HiddenMerger):
    def __init__(self, config):
        super().__init__(config)
        self.config=config
        num_layers=config.transformerscfg.num_hidden_layers
        self.expand=nn.Linear(config.transformerscfg.hidden_size*num_layers,config.mambacfg.d_model*num_layers)
        self.act=ACT2FN[config.transformerscfg.hidden_act]


    def forward(self,gist_activations):
        pass

class EffHiddenMerger(HiddenMerger):
    def __init__(self, config):
        super().__init__(config)
        self.config=config
        
        
        self.merge_proj=nn.Linear(config.mambacfg.d_model*2,config.mambacfg.d_model)

    def forward(self,gist_activations,hidden_states,prompt_input_idsm,idx):
        prompt_index=prompt_input_idsm.shape[-1]-1
        gist_indices=gist_activations.gist_indices
        gist_hiddens=gist_activations.all_last_hidden_state[:,idx,0:1,:]
        
        for i in range(hidden_states.shape[0]):
            
            hidden_state=hidden_states[i][prompt_index:prompt_index+1,:]
            
            expanded_hidden=gist_hiddens[i]
            dtype= expanded_hidden.dtype
           
            concat=torch.cat((hidden_state,expanded_hidden),1).to(dtype)

            hidden_states[i][prompt_index,:]=self.merge_proj(concat).to(hidden_states.dtype)

        return hidden_states
class ConMixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        conmambacfg:ConMambaConfig=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        
        
        
        self.merger=nn.ModuleList(
            [LastHiddenMerger(conmambacfg) for i in range(conmambacfg.transformerscfg.num_hidden_layers)]
        )
        num_layer_half=conmambacfg.transformerscfg.num_hidden_layers//2
        
        self.index=[2*i for i in range(num_layer_half)]+[2*i+1 for i in range(n_layer//2-num_layer_half,n_layer//2)]
        

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.gist_hidden=None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None,prompt_input_idsm:Optional[torch.LongTensor] = None,
        gist_activations:GistActivations=None,output_hidden_states: Optional[bool] = None,return_dict: Optional[bool] = None,
        inference=False
        
        ):
       
        hidden_states = self.embedding(input_ids)
        
        residual = None
        num=0
        
            
        for idx,layer in enumerate(self.layers):
            
            if inference :
                pass
            else:
                if idx in self.index:
                    hidden_states=self.merger[num](gist_activations,hidden_states,prompt_input_idsm,num)
                    
                    num+=1
            
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
        

       

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        return hidden_states

class CustomGenerationMixin(GenerationMixin):
    def __init__():
        super().__init__()


    def generate(
    self,
    add_tensor,
    max_length,
    top_k=1,
    top_p=0.0,
    min_p=0.0,
    temperature=1.0,
    return_dict_in_generate=False,
    output_scores=False,
    cg=True,
    gist_ids=None,
    prompt_input_idsm=None,
    attention_mask_gist=None,
    attention_mask=None,
    **kwargs,
    ):
        self.mamba.backbone.gist_hidden=None
        self.mamba.backbone.inference=True
        self.inference=True
        self.t=True

        
        
        output=super().generate(add_tensor=add_tensor,max_length=max_length,top_k=top_k,top_p=top_p,min_p=min_p,
                        temperature=temperature,return_dict_in_generate=return_dict_in_generate,output_scores=output_scores,cg=cg,**kwargs)
        self.mamba.backbone.inference=False
        self.inference=False
        
        return output

class ConMambaLMHeadModel(nn.Module, CustomGenerationMixin):

    def __init__(
        self,
        config: ConMambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        conmambacfg=config
        config=config.mambacfg
        
        self.config = config
        
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        
        self.backbone = ConMixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            conmambacfg=conmambacfg,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    # def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0,
    #     prompt_input_idsm:Optional[torch.LongTensor] = None,
    #     gist_activations:GistActivations=None, 
    #     gist_position: [int] =None,
    #     ):
    #     """
    #     "position_ids" is just to be compatible with Transformer generation. We don't use it.
    #     num_last_tokens: if > 0, only return the logits for the last n tokens
    #     """
    #     hidden_states = self.backbone(input_ids, inference_params=inference_params)
    #     if num_last_tokens > 0:
    #         hidden_states = hidden_states[:, -num_last_tokens:]
    #     lm_logits = self.lm_head(hidden_states)
    #     CausalLMOutput = namedtuple("CausalLMOutput", ["logits"])
    #     return CausalLMOutput(logits=lm_logits)
    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0,
        prompt_input_idsm:Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        gist_activations:GistActivations=None, 
        gist_position: [int] =None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inference=False,
        ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params,prompt_input_idsm=prompt_input_idsm,
        gist_activations=gist_activations,output_hidden_states=output_hidden_states,return_dict=return_dict,inference=inference,)
        
        return hidden_states

    @classmethod
    def from_pretrained(cls, pretrained_model_name, device="cpu",config=None,dtype=None, **kwargs):
        if config==None:
            config_data = load_config_hf(pretrained_model_name)
            config = MambaConfig(**config_data)
        
            
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype),strict=False)
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f)



COMABA_CLS = {
    "gistllama": GistLlama,
    "gistgemma": GistGemma,

}
class ConMambaForCausalLM(nn.Module,CustomGenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__()
        if config.from_pretrain:
            self.mamba = ConMambaLMHeadModel(config)
            
            self.transformers = COMABA_CLS[config._transformers_implementation](config.transformerscfg)
        self.lm_head = nn.Linear(config.mambacfg.d_model, config.mambacfg.vocab_size, bias=False)
        self.ll=nn.Linear(32,32)
        self.collator=None
        # Initialize weights and apply final processing
        # self.post_init()
        self.inference=False
        self.t=True
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, cache_params: Optional[MambaCache] = None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache_params"] = cache_params
        return model_inputs

    
    def forward(
        self,
        input_ids, position_ids=None, inference_params=None, num_last_tokens=0,
        gist_ids: Optional[torch.LongTensor] = None,
        prompt_input_idsm:Optional[torch.LongTensor] = None,
        attention_mask: torch.FloatTensor=None,
        attention_mask_gist: torch.FloatTensor=None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        gist_activations: Optional[GistActivations] = None,
        gist_offset: Optional[torch.LongTensor] = None,
        inference=False,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = True
        
        
        
        if inference :
            gist_act=None 
        else:
            model_outputs = self.transformers(
                    input_ids=gist_ids,
                    attention_mask=attention_mask,
                    attention_mask_gist=attention_mask_gist,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=return_dict,
                    gist_activations=gist_activations,
                    gist_offset=gist_offset,
                )
                
            gist_act=GistActivations.from_model_outputs(
                model_outputs=model_outputs,
                input_ids=gist_ids,
                gist_token=self.gist_token,
                num_gist_tokens=self.num_gist_tokens,
            )
        
        hidden_states = self.mamba(
            input_ids=input_ids,position_ids=position_ids,inference_params=inference_params,
            prompt_input_idsm=prompt_input_idsm,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            gist_activations=gist_act,
            use_cache=use_cache,
            inference=inference,
            
        )
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        if "ModulesToSaveWrapper" in str(self.lm_head):
            logits = self.lm_head(hidden_states.to(self.lm_head.original_module.weight.dtype)).float()
        else:
            logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + (hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )