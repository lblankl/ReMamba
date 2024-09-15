
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--basep", type=str, default=None)
args = parser.parse_args()
basep=args.basep
datap="result.json"
#read json 
import json
with open(basep+datap) as f:
    data = json.load(f)

#{'2wikimqa': {'0-4k': 19.39, '4-8k': 18.11, '8k+': 15.21}, 'gov_report': {'0-4k': 28.93, '4-8k': 25.18, '8k+': 24.5}, 'hotpotqa': {'0-4k': 25.21, '4-8k': 23.93, '8k+': 19.06}, 'lcc': {'0-4k': 60.51, '4-8k': 62.59, '8k+': 59.21}, 'multi_news': {'0-4k': 25.25, '4-8k': 20.24, '8k+': 20.63}, 'multifieldqa_en': {'0-4k': 39.46, '4-8k': 34.53, '8k+': 28.73}, 'passage_count': {'0-4k': 0.0, '4-8k': 0.0, '8k+': 0.0}, 'passage_retrieval_en': {'0-4k': 9.0, '4-8k': 5.0, '8k+': 11.0}, 'qasper': {'0-4k': 26.09, '4-8k': 14.6, '8k+': 12.33}, 'repobench-p': {'0-4k': 50.29, '4-8k': 48.88, '8k+': 45.42}, 'samsum': {'0-4k': 32.11, '4-8k': 30.02, '8k+': 31.96}, 'trec': {'0-4k': 51.0, '4-8k': 59.0, '8k+': 41.0}, 'triviaqa': {'0-4k': 64.16, '4-8k': 70.25, '8k+': 70.17}}
#avg the metrics on all seq length
import numpy as np
data_avg={}
for k,v in data.items():
    data_avg[k]=np.mean(list(v.values()))
print(data_avg)
#avg the the whole metrics
print(np.mean(list(data_avg.values())))
#add the avg to the dict
data_avg["avg"]=np.mean(list(data_avg.values()))
#save to json
with open(basep+"result_avg.json", "w") as f:
    json.dump(data_avg, f)
