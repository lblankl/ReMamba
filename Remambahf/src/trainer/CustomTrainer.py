from transformers import TrainingArguments,Trainer
class CustomerTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        #remove the  "label_ids" key value of the inputs.
        new_inputs = {k: v for k, v in inputs.items() if k != "label_ids"}
        outputs=model(**new_inputs)
        loss=outputs.loss
        return (loss, outputs) if return_outputs else loss