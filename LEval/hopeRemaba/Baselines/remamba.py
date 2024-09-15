import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
# Code adapted from https://huggingface.co/kaiokendev/superhot-13b-8k-no-rlhf-test/blob/main/llama_rope_scaled_monkey_patch.py

from functools import partial
import transformers
import torch

import argparse
from LEval_config import *
from tqdm import tqdm


from peft import PeftModel





from src.model.config_remamba import ReMambaConfig as ReMambaConfigfast
from src.model.configuration_ReMamba import ReMambaConfig
from mamba_ssm.utils.hf import load_config_hf
from src.model.ReMamba import ReMambaForCausalLM 
from src.model.ReMambafast import ReMambaLMHeadModel
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


# def replace_llama_with_condense(ratio):
#     transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = partial(
#         CondenseRotaryEmbedding, ratio=ratio, max_position=4096
#     )


def main(args):
    # openai.api_base = "https://api.openai-sb.com/v1"
    start_idx = 0
    device=args.device
    for file_name in key_data_pairs:
        fw = open(f'{file_name}', "w")
        data = key_data_pairs[file_name]
        header = (
            "A chat between a curious user and an artificial intelligence assistant."
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )

        sys_prompt = get_sys_prompt(args, file_name)
        sys_promptend=get_sys_prompt2(args, file_name)
        for d in tqdm(data):
            document = d['input']
           
            # while num_tokens_from_string(document, tokenizer) > max_length:
            #     if "code" not in file_name:
            #         document = " ".join(document.split(" ")[:max_length - cnt]) # chunk the input len from right
            #     else:
            #         document = " ".join(document.split(" ")[cnt - max_length:]) # chunk the input len from left
            #     cnt += 250

            instructions = d['instructions']
            outputs = d['outputs']

            for inst, out in zip(instructions, outputs):
                save_d = {}
                save_d['query'] = inst
                save_d['gt'] = out
                if "gsm" in file_name or "codeU" in file_name:
                    context = document + "\n\n" + sys_promptend +'\n'+inst
                    message = sys_prompt + context
                elif "topic" in file_name:
                    context = document + "\n\n" + sys_promptend +'\n'+ inst
                    message = sys_prompt + context
                    message += "\nAnswer:"
                elif args.metric == "exam_eval":
                    context = "Document is as follows. {document} "+"\n"+sys_promptend +'\n'+ " \nQuestion: {inst} "
                    message = sys_prompt + context
                    message += " \nAnswer:"
                elif "coursera" in file_name:
                    context = "Document is as follows. {document} "+"\n"+sys_promptend +'\n'+ " \nQuestion: {inst} "
                    message = sys_prompt + context + "\n Please only give the correct options (e.g., A)."
                    message += " \nAnswer:"
                else:
                    context = "Document is as follows. {document}"+"\n"+sys_promptend +'\n'+ "\nInstruction: {inst} " + f"The suggested output length is around {len(out.split())} words. "
                    message = sys_prompt + context
                    message += " \nMy english answer is:"
                try:
                    text_inputs = message.format(document=document, inst=inst)
                except:
                    text_inputs = message
                save_d['prompt'] = message.replace(document, "<long input>")
               
                tokenized_prompt = tokenizer(text_inputs, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
                if len(tokenized_prompt) > max_length:
                    half = int(max_length/2)
                    text_inputs = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            
                inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                prompt_length = inputs.input_ids.size()[-1]
                if 'fast' in args.model_type :
                    sample = model.generate(inputs.input_ids,temperature=0,max_length=max_new_tokens+prompt_length,eos_token_id=0,cg=True)[0]
                    
                else:
                    sample = model.generate(inputs.input_ids, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)[0]
                output = tokenizer.decode(sample[prompt_length:], skip_special_tokens=True)
                save_d[f'{open_source_model}_pred'] = output
                save_d['evaluation'] = d['evaluation']

                if "sci_fi" in file_name:
                    text_inputs = inst.replace("based on the world described in the document.", "based on the real-world knowledge and facts up until your last training") + "\nAnswer:"
                    inputs = tokenizer(text_inputs, return_tensors="pt").to(device)
                    if 'fast' in args.model_type :
                        sample = model.generate(inputs.input_ids,temperature=0,max_length=max_new_tokens+prompt_length,eos_token_id=0,cg=True)
                    else:
                        sample = model.generate(inputs.input_ids, do_sample=False, max_new_tokens=max_new_tokens)
                    prompt_length = inputs.input_ids.size()[-1]
                    output = tokenizer.decode(sample[0][prompt_length:])
                    save_d[f'{open_source_model}_pred'] += f" [fact: {output}]"

                if start_idx < 5:
                    print('document len', num_tokens_from_string(document, tokenizer))
                    print("----------------- [output] vs [ground truth] -----------------")
                    print('[output]:', save_d[f'{open_source_model}_pred'], "\n\n", '[ground truth]:', save_d['gt'])
                    start_idx += 1
                fw.write(json.dumps(save_d) + '\n')
        fw.close()
        # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    # set this if you do not want to use data from huggingface
    parser.add_argument('--task_path', type=str, default=None,
                        help='set this if you want test a specific task , example: LEval-data/Closed-ended-tasks/coursera.jsonl or LEval-data/Closed-ended-tasks/ ')
    # set this if you do not want to test a specific task
    parser.add_argument('--task_name', type=str, default=None,
                        help='set this if you want test a specific task from huggingface, example: coursera')
    parser.add_argument('--mc_tasks', action='store_true', help='set this if you want to test all multiple choice tasks')

    # for llama based model
    parser.add_argument('--scale', default='7b', choices=['7b', '13b'])
    parser.add_argument('--flash', action='store_true', help='set this if you want to use flash attention')


    parser.add_argument('--model_type', default=None, help='set this if you want to use a different model type')
    parser.add_argument('--model_name', default=None, help='set this if you want to use a different model name')
    parser.add_argument('--model_path', default=None, help='set this if you want to use a different model path')
    parser.add_argument('--peft_path', default=None, help='set this if you want to use a different peft path')
    parser.add_argument('--tokenizer', default=None, help='set this if you want to use a different tokenizer path')
    parser.add_argument("--remamba_config", type=str, default="../remamba/configs/config.json", help="The path to the Remamba config file.")
    parser.add_argument("--stratio",type=float, default=0.0, help="The stratio value for the PEFT algorithm.")
    parser.add_argument("--ratio",type=float, default=0.05, help="The ratio value for the PEFT algorithm.")
    parser.add_argument("--compressp_ratio",type=float, default=0.1, help="The compression ratio value for the PEFT algorithm.")

    parser.add_argument('--max_length', default="2k", help='max length of the input, e.g., 2k, 16k')
    parser.add_argument('--metric', choices=["llm_turbo_eval", "llm_gpt4_eval", "exam_eval", "ngram_eval", "human_eval"],
                        help='metric name from choices', required=True)
    parser.add_argument("--base_path", type=str, default=None, help="The path ")

    args = parser.parse_args()

    # 7b / 13b

    max_length = k_to_number(args.max_length) - max_new_tokens

   
   
    open_source_model = args.model_name


    # if args.flash:
    #     replace_llama_attn_with_flash_attn()


    data_save_path = args.base_path+"/"+f"Predictions/{args.metric}/{open_source_model}"
    # input(f"Your prediction file will be saved to: {data_save_path}  , press enter to confirm...")

    device = args.device
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)

    model_type = args.model_type
    if model_type == "ReMambafast":
        config_data = load_config_hf(args.model_path)
        cfg = ReMambaConfigfast(**config_data)
        cfg.ratio=args.ratio
        cfg.stratio=args.stratio
        cfg.compressp_ratio=args.compressp_ratio
        model = ReMambaLMHeadModel.from_pretrained(args.model_path, config=cfg, dtype=torch.bfloat16)
    elif model_type == "ReMamba":
        cfg=ReMambaConfig.from_pretrained(args.remamba_config)
        cfg.ratio=args.ratio
        cfg.stratio=args.stratio
        cfg.compressp_ratio=args.compressp_ratio
        model = ReMambaForCausalLM.from_pretrained(args.model_path, config=cfg, trust_remote_code=True,torch_dtype=torch.bfloat16)
    elif model_type == "mamba":
        model=AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,torch_dtype=torch.bfloat16)
        
    elif model_type == "mamba2" or model_type == "mamba2fast" or model_type == "mambafast":
        model = MambaLMHeadModel.from_pretrained(model_type, dtype=torch.bfloat16)
    else:
        model=AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True,torch_dtype=torch.bfloat16)

    if args.peft_path and args.peft_path!="None":
        model= PeftModel.from_pretrained(model,args.peft_path).merge_and_unload()
    
    model=model.to(device)
    model.eval()
    key_data_pairs = {}
    build_key_data_pairs(args, key_data_pairs, data_save_path)
    sys.exit(main(args))
