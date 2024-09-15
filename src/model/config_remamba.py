from dataclasses import dataclass, field
from transformers import PretrainedConfig

# @dataclass
# class ReMambaConfig(PretrainedConfig):

#     d_model: int = 2560
#     d_intermediate: int = 0
#     n_layer: int = 64
#     vocab_size: int = 50277
#     ssm_cfg: dict = field(default_factory=dict)
#     attn_layer_idx: list = field(default_factory=list)
#     attn_cfg: dict = field(default_factory=dict)
#     rms_norm: bool = True
#     residual_in_fp32: bool = True
#     fused_add_norm: bool = True
#     pad_vocab_size_multiple: int = 8
#     tie_embeddings: bool = True
#     ratio=1/10
#     compressp_ratio=0.3
#     stratio=0.0

from dataclasses import dataclass, field
from transformers import PretrainedConfig

# @dataclass
# class ReMambaConfig(PretrainedConfig):

#     d_model: int = 2560
#     d_intermediate: int = 0
#     n_layer: int = 64
#     vocab_size: int = 50277
#     ssm_cfg: dict = field(default_factory=dict)
#     attn_layer_idx: list = field(default_factory=list)
#     attn_cfg: dict = field(default_factory=dict)
#     rms_norm: bool = True
#     residual_in_fp32: bool = True
#     fused_add_norm: bool = True
#     pad_vocab_size_multiple: int = 8
#     tie_embeddings: bool = True
#     ratio=1/10
#     compressp_ratio=0.3
#     stratio=0.0
    
class ReMambaConfig(PretrainedConfig):
    def __init__(self, d_model=2560, d_intermediate=0, n_layer=64, vocab_size=50277, ssm_cfg={}, 
                 attn_layer_idx=[], attn_cfg={}, rms_norm=True, residual_in_fp32=True, fused_add_norm=True,
                   pad_vocab_size_multiple=8, tie_embeddings=True, ratio=1/10, compressp_ratio=0.3, stratio=0.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.attn_layer_idx = attn_layer_idx
        self.attn_cfg = attn_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_embeddings = tie_embeddings
        self.ratio = ratio
        self.compressp_ratio = compressp_ratio
        self.stratio = stratio
        self.update(kwargs)
        super().__init__(**kwargs)