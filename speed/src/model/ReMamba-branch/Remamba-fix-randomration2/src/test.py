from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig,get_peft_model,PeftModel
from datasets import load_dataset
from src.dataset.collator import DataCollatorForConMamba
from src.model import GistLlama
import torch
from src.dataset.preprocessg import Preprocess as Preprocess
class ModelArguments:
    def __init__(self):
      
        self.logging_dir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/log/llama7b"
        self.tokenizer= "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/meta/llama7b-chat"
        self.mtokenizer= "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8btemplate"

        self.modelpath = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Model/General/llama-2-7b"
        self.output_dir= "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/ConMamba/out/llama7b"
        self.lora=32
args=ModelArguments()


tokenizer=AutoTokenizer.from_pretrained(args.tokenizer)
tokenizer.pad_token_id=0
model=GistLlama.GistLlama.from_pretrained(args.modelpath)

if len(tokenizer) == 32000 + 1:
    pass
else:
    # Initialize gist token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<GIST>"]})
    model.resize_token_embeddings(len(tokenizer))
    # Set new word embedding to average of existing word embeddings. For why,
    # see https://nlp.stanford.edu/~johnhew/vocab-expansion.html
    
    with torch.no_grad():
            
                model.model.embed_tokens.weight[
                    -1
                ] = model.model.embed_tokens.weight[:-1].mean(0)
                model.lm_head.weight[-1] = model.lm_head.weight[:-1].mean(0)
          
gist_token = tokenizer.additional_special_tokens_ids[-1]

dp="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/concept/conceptTransformer248/datasets/OpenOrca"
dataset = load_dataset(dp, split="train").select(range(1))
mtokenizer=AutoTokenizer.from_pretrained(args.mtokenizer)
collator=DataCollatorForConMamba(
            tokenizer,
            mtokenizer,
            max_length=256 + 256,  # source=256; target=256
            # Human eval examples are longer.
            max_length_human=384 + 384,  # source=384; target=384
            gist_condition="gist",
            num_gist_tokens=1,
            gist_token=gist_token,
            pad_token=tokenizer.pad_token_id,
            check_correctness=True,
        )
preprocess=Preprocess(tokenizer)

ndataset=dataset.map(preprocess,remove_columns=["system_prompt","question","response","id"])
d=collator(ndataset)
torch.set_printoptions(profile="full")
print(d['attention_mask_gist'][0][0])
print(d['input_ids'])
print(tokenizer.decode(d['input_ids'][0]))