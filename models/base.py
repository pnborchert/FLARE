import torch, os, glob
from transformers import PreTrainedModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Outputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Tuple[torch.FloatTensor] = None

class CLS(PreTrainedModel):
    def __init__(self, encoder, args):
        super().__init__(encoder.config)

        self.config = encoder.config
        self.encoder = encoder
        
        self.loss_fct_ce = torch.nn.CrossEntropyLoss()
    
    def load_from_ckpt(self, ckpt_path):

        print("*"*20, f"Loading from {ckpt_path}", "*"*20)

        state_dict = {}
        for file in glob.glob(os.path.join(ckpt_path, "*pytorch_model*.bin")):
            state_dict.update(torch.load(file, map_location="cuda"))

        self.load_state_dict(state_dict, strict=False)
    
    def forward(self, input_ids, attention_mask, labels, **kwargs):
        return self.encoder(input_ids.long(), attention_mask=attention_mask.long(), labels=labels.long(), output_hidden_states=False)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # labels = inputs.pop("labels")
        out = model(**inputs, is_eval=not model.training)
        loss = out.loss

        return (loss, out) if return_outputs else loss

class Decoder(PreTrainedModel):
    def __init__(self, encoder, args):
        super().__init__(encoder.config)

        self.config = encoder.config
        self.encoder = encoder
        self.generate_max_tokens = args["generate_max_tokens"]
            
    def load_from_ckpt(self, ckpt_path):

        print("*"*20, f"Loading from {ckpt_path}", "*"*20)

        state_dict = {}
        for file in glob.glob(os.path.join(ckpt_path, "*pytorch_model*.bin")):
            state_dict.update(torch.load(file, map_location="cuda"))

        self.load_state_dict(state_dict, strict=False)
    
    def forward(self, input_ids, attention_mask, labels, **kwargs):
        outputs = self.encoder(input_ids.long(), attention_mask=attention_mask.long(), labels=labels.long(), output_hidden_states=False)
        return outputs

    def generate(self, input_ids, attention_mask, **kwargs):
        return self.encoder.generate(input_ids=input_ids.long(), attention_mask=attention_mask.long(), max_new_tokens=self.generate_max_tokens, output_hidden_states=False, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels = labels.long()

        is_eval = not model.training
        out = model(**inputs, labels=labels, is_eval=is_eval)
        
        loss = out.loss

        out = Outputs(
            loss=loss,
            logits=out.logits.unsqueeze(0),
        )

        return (loss, out) if return_outputs else loss
    
class QA(PreTrainedModel):
    def __init__(self, encoder, args):
        super().__init__(encoder.config)

        self.config = encoder.config
        self.encoder = encoder
    
    def load_from_ckpt(self, ckpt_path):
        print("*"*20, f"Loading from {ckpt_path}", "*"*20)

        state_dict = {}
        for file in glob.glob(os.path.join(ckpt_path, "*pytorch_model*.bin")):
            state_dict.update(torch.load(file, map_location="cuda"))
        
        self.load_state_dict(state_dict, strict=False) 
            
    def forward(self, input_ids, attention_mask, start_positions, end_positions, **kwargs):
        return self.encoder(input_ids=input_ids.long(), attention_mask=attention_mask.long(), start_positions=start_positions, end_positions=end_positions, output_hidden_states=False)
    
    def compute_loss(self, model, inputs, return_outputs=False):

        is_eval = not model.training
        out = model(**inputs, is_eval=is_eval)

        loss = out.loss

        return (loss, out) if return_outputs else loss
    
class QADecoder(PreTrainedModel):
    def __init__(self, encoder, args):
        super().__init__(encoder.config)

        self.config = encoder.config
        self.encoder = encoder
        self.generate_max_tokens = args["generate_max_tokens"]

    def load_from_ckpt(self, ckpt_path):
        print("*"*20, f"Loading from {ckpt_path}", "*"*20)

        state_dict = {}
        for file in glob.glob(os.path.join(ckpt_path, "*pytorch_model*.bin")):
            state_dict.update(torch.load(file, map_location="cuda"), strict=False)
    
    def forward(self, input_ids, attention_mask, labels, **kwargs):
        outputs = self.encoder(input_ids.long(), attention_mask=attention_mask.long(), labels=labels.long(), output_hidden_states=False)
        return outputs

    def generate(self, input_ids, attention_mask, **kwargs):
        return self.encoder.generate(input_ids=input_ids.long(), attention_mask=attention_mask.long(), max_new_tokens=self.generate_max_tokens, output_hidden_states=False, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        labels = labels.long()

        # replace padding with -100
        labels = torch.where(labels == self.config.pad_token_id, torch.tensor(-100).to(labels.device), labels)
        
        is_eval = not model.training
        out = model(**inputs, labels=labels, is_eval=is_eval)
        
        loss = out.loss

        out = Outputs(
            loss=loss,
            logits=out.logits.unsqueeze(0),
        )

        return (loss, out) if return_outputs else loss