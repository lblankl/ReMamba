
import gradio as gr

m13="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujiahao12/Model/General/llama-2-13b"
m70="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/model/Qwen1.5-72B-Chat"
mp=m70
from transformers import AutoTokenizer
from transformers import LlamaConfig
import torch

from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights
model=AutoModelForCausalLM.from_pretrained(mp,device_map="auto",torch_dtype=torch.float16)

examples = [
    [[],0.9,100],
    [[],0.8,100]
    
]

tp="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/yuandl/LLMs/QWen/Qwen1.5-72c"
tokenizer=AutoTokenizer.from_pretrained(tp)
from transformers import pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
def process(messages,temperature,max_tokens):

    text=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
     
    if temperature==0:
        res=generator(text, max_length=max_tokens, do_sample=False,return_full_text=False)
    else:
        res=generator(text, max_length=max_tokens, do_sample=True,temperature=temperature,return_full_text=False)
    
    res=res[0]['generated_text']
    return res


iface = gr.Interface(  
    fn=process,  
    inputs=[gr.JSON(label="请输入或使用下面的examples:"),gr.Number(label="temperature"),gr.Number(label="max_tokens")],  
    outputs=gr.Textbox(),
    title="展示demo名称",
    description="输入文本，分析。注意每次输入前请清空当前输入。",
    examples=examples, 
    allow_flagging="never",
)

iface.launch(debug=True, share=False, server_name="0.0.0.0", server_port=7860)

# from gradio_client import Client
 
# client = Client("https://1f29586e-0bdd-423d-a823-644f7dcbc479-vscode-zw05.mlp.sankuai.com/proxy/7860")
# print(client.predict("22"))