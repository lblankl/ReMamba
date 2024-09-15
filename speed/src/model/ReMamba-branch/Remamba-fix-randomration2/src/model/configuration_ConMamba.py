
from transformers import LlamaConfig,GemmaConfig,PretrainedConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.models.config_mamba import MambaConfig
def cfg(dic):
    return MambaConfig(**dic)
class ConMambaConfig(PretrainedConfig):
    

    model_type = "conmamba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        _transformers_implementation="gistllama",
        mambacfg=cfg(load_config_hf("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-nohf-Sim")),
        transformerscfg=LlamaConfig.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/openlm-research/open_llama_3b_v2"),
        from_pretrain=False,
        **kwargs,
    ):
        self._transformers_implementation=_transformers_implementation
        self.mambacfg=mambacfg
        self.transformerscfg=transformerscfg
        self.from_pretrain=from_pretrain
        self.hidden_size=self.transformerscfg.hidden_size
        super().__init__(**kwargs)

       