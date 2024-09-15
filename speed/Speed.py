
from transformers import AutoConfig,LlamaForCausalLM


from mamba_ssm.models.config_mamba import MambaConfig
from src.model.ReMambafast import ReMambaLMHeadModel
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf

from src.model.config_remamba import ReMambaConfig 

import torch
import time
basepath="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/speed/res"
debug=False
device1="cuda:0"
device2="cuda:0"
device3="cuda:0"
llamacfg="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/openlm-research/open_llama_3b_v2"
mambacfg="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-nohf"
mamba2cfg="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba2-2.7b-nohf"

config_datamamba1 = load_config_hf(mambacfg)
remamba1cfg=ReMambaConfig(**config_datamamba1)
mamba1cfg=MambaConfig(**config_datamamba1)

config_datamamba2 = load_config_hf(mamba2cfg)
remamba2cfg=ReMambaConfig(**config_datamamba2)
mamba2cfg=MambaConfig(**config_datamamba2)


llamaconfig=AutoConfig.from_pretrained(llamacfg)


if debug:
    
    remamba1cfg.d_model=128
    remamba1cfg.n_layer=4
    mamba1cfg.d_model=128
    mamba1cfg.n_layer=4

    remamba2cfg.d_model=256
    remamba2cfg.n_layer=4
    mamba2cfg.d_model=256
    mamba2cfg.n_layer=4

    # llamaconfig.hidden_size=128
    # llamaconfig.num_hidden_layers=4
    # llamaconfig.num_attention_heads=8
    # llamaconfig.intermediate_size=512



remamba1=ReMambaLMHeadModel(remamba1cfg).to(device1)
mamba1=MambaLMHeadModel(mamba1cfg).to(device2)
remamba2=ReMambaLMHeadModel(remamba2cfg).to(device1)
mamba2=MambaLMHeadModel(mamba2cfg).to(device2)
llama=LlamaForCausalLM(llamaconfig).to(device3)


import time 
model_dic={"mamba1":mamba1,"remamba1":remamba1,"mamba2":mamba2,"remamba2":remamba2,"llama3b":llama}
# if debug:
#     model_dic={"mamba2.8b":mamba,"remamba2.8b":remamba}
if debug:
    inp_len_list=[128,256]
    out_len_list=[128,256]
else:
    inp_len_list=[1024,2048,4096,6144,8192]
    out_len_list=[512,1024]

time_dic={}
speed_dic={}

for model_name,model in model_dic.items():
    if 'mamba' in model_name:
        print("warm up ",model_name)
        #mamba needs to warm up first
        warmp_up_input_ids=[5]*128
        warm_up_inputs=torch.tensor(warmp_up_input_ids).unsqueeze(0).to(next(model.parameters()).device)
        warm_up_o=model.generate(warm_up_inputs,max_length=136,temperature=1,cg=True)
    time_dic[model_name]={}
    speed_dic[model_name]={}
    for inp_len in inp_len_list:
        for out_len in out_len_list:
            
            input_ids=[5]*inp_len
            inputs=torch.tensor(input_ids).unsqueeze(0).to(next(model.parameters()).device)
            attention_mask=torch.ones_like(inputs)
            if 'mamba' in model_name:
               
                start=time.time()
                
                o=model.generate(inputs,max_length=out_len+inp_len,temperature=1,cg=True)
                end=time.time()
            else:
                start=time.time()
                
                o=model.generate(inputs,max_length=out_len+inp_len,do_sample=False)
                end=time.time()
            timespent=end-start
            time_dic[model_name][str(inp_len)+"_"+str(out_len)]=timespent
            speed_dic[model_name][str(inp_len)+"_"+str(out_len)]=out_len/timespent
            print(f"model:{model_name},inp_len:{inp_len},out_len:{out_len},time:{end-start},speed:{out_len/(end-start)}")
print(time_dic)
print(speed_dic)
import json
with open(basepath+"/time.json","w") as f:
    json.dump(time_dic,f)
#turn into a table
import pandas as pd
df=pd.DataFrame(time_dic).T
df.to_csv(basepath+"/time.csv")

df=pd.DataFrame(speed_dic).T
df.to_csv(basepath+"/speed.csv")
