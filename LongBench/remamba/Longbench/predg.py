import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse






from src.model.ReMamba import ReMambaForCausalLM 
from src.model.ReMambafast import ReMambaLMHeadModel
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from src.model.configuration_ReMambahf import ReMambaConfig as ReMambaConfighf
from src.model.configuration_ReMamba import ReMambaConfig as  ReMambaConfigfast



from mamba_ssm.utils.hf import load_config_hf



# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/LongBench/THUDM/LongBench')
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--out_name', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--cfg_path', type=str, default=None)
    parser.add_argument('--remamba_sample_ratio', type=float, default=0.1)
    parser.add_argument('--stratio', type=float, default=0.0)
    parser.add_argument('--compressp_ratio', type=float, default=0.3)
    parser.add_argument('--append_prompt', type=str,default='False', help="Evaluate on LongBench-E")
    parser.add_argument('--peft_path', type=str, default=None)
    parser.add_argument('--max_len', type=int, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    
    
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    # elif "llama2" in model_name:
    #     prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    else:
        messages = [
        {
            "role": "system",
            "content": "You are a helpful ai assistant to fullfill users' needs.",
        },
        {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format,prompt_formatap, dataset, device, model_name, model2path, out_path,peftpath,tokenpath,args):
    device = torch.device(f'cuda:{rank}')
    model, tokenizer = load_model_and_tokenizer(model2path,peftpath[model_name],tokenpath, model_name, device,args)
    # if model_name=="conmamba":
    #     ttokenizer=AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/openlm-research/open_llama_3b_v2template")
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        if prompt_formatap is not None:
            promptap = prompt_formatap.format(**json_obj)
        else:
            promptap=None
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        
       
        
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
       
        if "chatglm3" in model_name:
            if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            # if flag==1:
            #     input = tokenizer(prompt, truncation=False, return_tensors="pt",add_special_tokens=False).to(device)
            # else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            if promptap is not None:
                ap= tokenizer.encode(promptap, truncation=False, return_tensors="pt",add_special_tokens=False).to(device)
                model.backbone.prompt_ids=ap
        context_length = input.input_ids.shape[-1]
       
        if "mamba" in model_name.lower():
            input.pop('attention_mask')
        if dataset == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            if 'fast' in model_name.lower():
                output = model.generate(input.input_ids,temperature=0,max_length=max_gen+context_length,eos_token_id=0,cg=True)[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                    min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
        else:
            if 'fast' in model_name.lower():
                output = model.generate(input.input_ids,temperature=0,max_length=max_gen+context_length,eos_token_id=0,cg=True)[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]
        
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        
        pred = post_process(pred, model_name)
        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]}, f, ensure_ascii=False)
            f.write('\n')
    dist.destroy_process_group()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
from peft import PeftModel
def load_model_and_tokenizer(path,peftpath,tokenpath, model_name, device,args):
    tokenizer = AutoTokenizer.from_pretrained(tokenpath, trust_remote_code=True)
   
    if model_name == "ReMamba":
        cfg=ReMambaConfighf.from_pretrained(args.cfg_path)
        cfg.ratio=args.remamba_sample_ratio
        cfg.stratio=args.stratio
        cfg.compressp_ratio=args.compressp_ratio
        model = ReMambaForCausalLM.from_pretrained(args.model_path, config=cfg, trust_remote_code=True,torch_dtype=torch.bfloat16)
  
    elif model_name == "ReMambafast" or model_name == "ReMamba2fast" or model_name == "ReMamba2":
        config_data = load_config_hf(args.model_path)
        cfg = ReMambaConfigfast(**config_data)
        cfg.ratio=args.remamba_sample_ratio
        cfg.stratio=args.stratio
        cfg.compressp_ratio=args.compressp_ratio
        model = ReMambaLMHeadModel.from_pretrained(args.model_path, config=cfg, dtype=torch.bfloat16)
    elif model_name == "mamba2" or model_name == "mamba2fast" or model_name == "mambafast":
        model = MambaLMHeadModel.from_pretrained(args.model_path, dtype=torch.bfloat16)
    elif model_name == "mamba":
        model=AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        print('normal')
    if peftpath is None or peftpath =='None':
        print('base empty')
        model=model.to(device)
    else:
        model=PeftModel.from_pretrained(model,peftpath).merge_and_unload().to(device)
    
   
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)

    model2path = args.model_path
    if args.max_len:
      
        max_length = args.max_len
    model_name = args.model
    if args.peft_path:
        print('custom peftp')
        peftpath={model_name:args.peft_path}
   
    else:
        print("empty")
        peftpath=""
    tokenpath=args.tokenizer
    if tokenpath is None:
        tokenpath=args.model_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfgp=args.cfg_path
    
    
    
  
   
    
    if args.e:
        print("en")
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    if args.task:
        datasets=[args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2promptap=json.load(open("config/dataset2promptap.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    if args.out_name:
        out_name=args.out_name
    else:
        out_name=model_name
    for dataset in datasets:
        if args.e:
            data = load_dataset(args.datapath, f"{dataset}_e", split='test')
            if not os.path.exists(f"pred_e/{out_name}"):
                os.makedirs(f"pred_e/{out_name}")
            out_path = f"pred_e/{out_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            if not os.path.exists(f"pred/{out_name}"):
                os.makedirs(f"pred/{out_name}")
            out_path = f"pred/{out_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        if args.append_prompt=='True':
            prompt_formatap=dataset2promptap[dataset]
        else:
            prompt_formatap=None
        data_all = [data_sample for data_sample in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(rank, world_size, data_subsets[rank], max_length, \
                        max_gen, prompt_format,prompt_formatap, dataset, device, model_name, model2path, out_path,peftpath,tokenpath,args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
