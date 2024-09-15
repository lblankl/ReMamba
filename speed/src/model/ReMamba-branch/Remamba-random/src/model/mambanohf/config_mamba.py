# from dataclasses import dataclass, field
# from transformers import PretrainedConfig

# @dataclass
# class MambaConfig(PretrainedConfig):

#     d_model: int = 2560
#     n_layer: int = 64
#     vocab_size: int = 50277
#     ssm_cfg: dict = field(default_factory=dict)
#     rms_norm: bool = True
#     residual_in_fp32: bool = True
#     fused_add_norm: bool = True
#     pad_vocab_size_multiple: int = 8
#     tie_embeddings: bool = True
#     hidden_size:int =2560
from dataclasses import dataclass, field
from transformers import PretrainedConfig

@dataclass
class MambaConfig(PretrainedConfig):

    # d_model: int = 2560
    # n_layer: int = 64
    # vocab_size: int = 50277
    # ssm_cfg: dict = field(default_factory=dict)
    # rms_norm: bool = True
    # residual_in_fp32: bool = True
    # fused_add_norm: bool = True
    # pad_vocab_size_multiple: int = 8
    # tie_embeddings: bool = True
    def __init__(
        self,
        d_model: int = 2560,
        n_layer: int = 64,
        vocab_size: int = 50277,
        ssm_cfg: dict = {},
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        pad_vocab_size_multiple: int = 8,
        tie_embeddings: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_embeddings = tie_embeddings
        self.hidden_size = d_model
