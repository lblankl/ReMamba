
from transformers import LlamaConfig,MambaConfig,GemmaConfig,PretrainedConfig,AutoConfig

class ConMambaConfig(PretrainedConfig):
    

    model_type = "conmamba"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        _transformers_implementation="gistllama",
        mambacfg=MambaConfig.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-hf"),
        transformerscfg=LlamaConfig.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Model/General/llama-2-7b"),
        from_pretrain=False,
        **kwargs,
    ):
        self._transformers_implementation=_transformers_implementation
        
        if type(mambacfg)==dict:
            self.mambacfg=MambaConfig.from_dict(mambacfg)
        else:
            self.mambacfg=mambacfg
        if type(transformerscfg)==dict:
            self.transformerscfg=LlamaConfig.from_dict(transformerscfg)
        else:
            self.transformerscfg=transformerscfg
        self.from_pretrain=from_pretrain
        self.hidden_size=self.transformerscfg.hidden_size
        super().__init__(**kwargs)

       