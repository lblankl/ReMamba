from datasets import load_dataset
from transformers import AutoTokenizer
from .dataset.Orca import OrcaPreprocess
from .dataset.Long import LongPreprocess
from datasets import concatenate_datasets
import argparse


parser=argparse.ArgumentParser()
parser.add_argument("--tokenizer",default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/state-spaces/mamba-2.8b-hf")
parser.add_argument("--longdp",default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/LongAlpaca-12k")
parser.add_argument("--Orcadp",default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/concept/conceptTransformer248/datasets/OpenOrca")
parser.add_argument("--debug",default=False,help="debug mode")
parser.add_argument("--maxlen",type=int,default=6000,help="max len of the context")

parser.add_argument("--normal_datapath",default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/NormalOrcaRemamba6000")

parser.add_argument("--long_datapath",default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/dataset/LongContextRemamba6000")

args=parser.parse_args()

mamba_tokenizer=AutoTokenizer.from_pretrained(args.tokenizer)
debug=args.debug
maxlen=args.maxlen
#Fist load Orca dataset 
Orca=load_dataset(args.Orcadp,split='train')

if debug:
    print("debug:Orca")
    Orca=Orca.select(range(1000))

OrcaPreprocessor=OrcaPreprocess(mamba_tokenizer, max_len=maxlen)    

# this First preprocess Orca dataset. Truncate the context to max len.But just for the context, not the response.
#Which means the response will not be truncated.Still some may exceed the max len.
Orca=Orca.map(OrcaPreprocessor, remove_columns=["system_prompt","question","response","id"])

#get thoes examples of length between 1k-6k  and save them to a new dataset
LongOrca=Orca.filter(lambda x: x['len_of_prompt']>700 and len(x['input_ids'])<maxlen+10)

print("LongOrca len:",len(LongOrca))

# Moreover get 500k normal examples but we just use 300k in training.
if debug:
    NormalOrca=Orca.select(range(1000))
else:
    NormalOrca=Orca.select(range(500000))
NormalOrca=NormalOrca.filter(lambda x: len(x['input_ids'])<maxlen)
# then load LongAlpaca dataset
LongAlpaca=load_dataset(args.longdp,split='train')

if debug:
    print("debug:LongAlpaca")
    LongAlpaca=LongAlpaca.select(range(1000))
LongAlpacaPreprocessor=LongPreprocess(mamba_tokenizer, max_len=maxlen)
LongAlpaca=LongAlpaca.map(LongAlpacaPreprocessor, remove_columns=['input', 'file', 'instruction', 'output'])
#just get those length below 6k and prompt length above 1k
LongAlpaca=LongAlpaca.filter(lambda x: x['len_of_prompt']>700 and len(x['input_ids'])<maxlen+10)

#Concatenate the two datasets: LongAlpaca and LongOrca as the LongContext dataset
LongContextdataset=concatenate_datasets([LongAlpaca,LongOrca])
print("LongContextdataset len:",len(LongContextdataset))
# #Concatenate the two datasets: NormalOrca and LongContextdataset as the final dataset
# Finaldataset=concatenate_datasets([NormalOrca,LongContextdataset])
#save them separately
LongContextdataset=LongContextdataset.shuffle()
NormalOrca=NormalOrca.shuffle()


NormalOrca.save_to_disk(args.normal_datapath)
LongContextdataset.save_to_disk(args.long_datapath)


