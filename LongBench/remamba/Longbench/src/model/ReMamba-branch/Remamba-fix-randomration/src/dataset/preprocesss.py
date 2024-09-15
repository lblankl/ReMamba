
class Preprocess():
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
        


    def __call__(self,examples):
        """"Preprocess the data for the model .Do not add special tokens at the beginning of the response. 
            input_for_concept: the inputids for the concept encoder (large llama model)  contain the prompt+response with last semantic seperation truncated : truncated(systemprompt+question+response)
            input_for_language: the inputids for the language encoder (small llama model)  contain the response with eos token. if response_token is not empty, the content is response_token+response+eos
            mask_lan_concept: the mask for the language decoder tensor with respect to the concept encoder tensor of shape (len(input_for_language),len(input_for_concept))

        """
        systemprompt=examples["system_prompt"]
        question=examples["question"]
        response=examples["response"]
        
        messages = [
        {
            "role": "system",
            "content": systemprompt,
        },
        {"role": "user", "content": question},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        promptlen=len(self.tokenizer.encode(prompt,add_special_tokens=False))
        resonselen=len(self.tokenizer.encode(response,add_special_tokens=False))+1
        
        
            
        examples["input"]=prompt
        examples["output"]=response
        examples["length"]=promptlen+resonselen
       
        
        return examples