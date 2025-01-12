import torch, glob, os
from .fusion import CLS, QA, QADecoder, Decoder

class FuseMTCLS(CLS):
    def __init__(self, encoder, args):
        super().__init__(encoder, args)
        
        self.is_enc_dec = args["plm"] in ["mt5-xl"]
        self.mt_dim = args["mt_model_dim"]
        self.encoder_dim = self.encoder.config.hidden_size
        self.n_enc_layers = encoder.config.num_hidden_layers

        self.proj = torch.nn.Linear(self.mt_dim, self.encoder_dim)
    
    def load_from_ckpt(self, ckpt_path):

        print("*"*20, f"Loading from {ckpt_path}", "*"*20)

        state_dict = {}
        for file in glob.glob(os.path.join(ckpt_path, "*pytorch_model*.bin")):
            state_dict.update(torch.load(file, map_location="cuda"))

        self.load_state_dict(state_dict, strict=False)

    def forward(self, input_ids, attention_mask, x_source, labels, **kwargs):
        # project to encoder hidden size
        x_source = self.proj(x_source) # (B, SEQ, D_enc)
        
        # tuple with n_enc_layers tensors of (B, SEQ, D_enc)
        x_source = tuple([x_source for _ in range(self.n_enc_layers)])

        if self.is_enc_dec:
            x_source = {"x_source_encoder": x_source,"x_source_decoder": x_source,}
        else:
            x_source = {"x_source": x_source}

        return self.encoder(input_ids.long(), attention_mask=attention_mask.long(), labels=labels, output_hidden_states=False, **x_source)
    
class FuseMTDecoder(Decoder):
    def __init__(self, encoder, args):
        super().__init__(encoder, args)
        
        self.mt_dim = args["mt_model_dim"]
        self.encoder_dim = self.encoder.config.hidden_size
        self.n_enc_layers = encoder.config.num_hidden_layers

        self.proj = torch.nn.Linear(self.mt_dim, self.encoder_dim)
    
    def load_from_ckpt(self, ckpt_path):

        print("*"*20, f"Loading from {ckpt_path}", "*"*20)

        state_dict = {}
        for file in glob.glob(os.path.join(ckpt_path, "*pytorch_model*.bin")):
            state_dict.update(torch.load(file, map_location="cuda"))

        self.load_state_dict(state_dict, strict=False)

    def forward(self, input_ids, attention_mask, x_source, labels, **kwargs):

        # project to encoder hidden size
        x_source = self.proj(x_source) # (B, SEQ, D_enc)
        
        # tuple with n_enc_layers tensors of (B, SEQ, D_enc)
        x_source = tuple([x_source for _ in range(self.n_enc_layers)])

        x_source = {"x_source": x_source}

        return self.encoder(input_ids.long(), attention_mask=attention_mask.long(), labels=labels.long(), output_hidden_states=False, **x_source)
    
    def generate(self, input_ids, attention_mask, x_source, **kwargs):

        # project to encoder hidden size
        x_source = self.proj(x_source)

        # tuple with n_enc_layers tensors of (B, SEQ, D_enc)
        x_source = [tuple([x_source for _ in range(self.n_enc_layers)])]

        x_source = {"x_source": x_source}

        return self.encoder.generate(input_ids=input_ids.long(), attention_mask=attention_mask.long(), max_new_tokens=self.generate_max_tokens, output_hidden_states=False, **x_source)

class FuseMTQA(QA):
    def __init__(self, encoder, args):
        super().__init__(encoder, args)
        
        self.mt_dim = args["mt_model_dim"]
        self.encoder_dim = self.encoder.config.hidden_size
        self.n_enc_layers = encoder.config.num_hidden_layers

        self.proj = torch.nn.Linear(self.mt_dim, self.encoder_dim)
    
    def load_from_ckpt(self, ckpt_path):

        print("*"*20, f"Loading from {ckpt_path}", "*"*20)

        state_dict = {}
        for file in glob.glob(os.path.join(ckpt_path, "*pytorch_model*.bin")):
            state_dict.update(torch.load(file, map_location="cuda"))

        self.load_state_dict(state_dict, strict=False) 

    def forward(self, input_ids, attention_mask, start_positions, end_positions, x_source, **kwargs):
        # project to encoder hidden size
        x_source = self.proj(x_source) # (B, SEQ, D_enc)
        
        # tuple with n_enc_layers tensors of (B, SEQ, D_enc)
        x_source = tuple([x_source for _ in range(self.n_enc_layers)])

        x_source = {"x_source": x_source}

        return self.encoder(input_ids.long(), attention_mask=attention_mask.long(), start_positions=start_positions, end_positions=end_positions, output_hidden_states=False, **x_source)

class FuseMTQADecoder(QADecoder):
    def __init__(self, encoder, args):
        super().__init__(encoder, args)
        
        self.mt_dim = args["mt_model_dim"]
        self.encoder_dim = self.encoder.config.hidden_size
        self.n_enc_layers = encoder.config.num_hidden_layers

        self.proj = torch.nn.Linear(self.mt_dim, self.encoder_dim)
    
    def load_from_ckpt(self, ckpt_path):

        print("*"*20, f"Loading from {ckpt_path}", "*"*20)

        state_dict = {}
        for file in glob.glob(os.path.join(ckpt_path, "*pytorch_model*.bin")):
            state_dict.update(torch.load(file, map_location="cuda"))

        self.load_state_dict(state_dict, strict=False)

    def forward(self, input_ids, attention_mask, x_source, labels, **kwargs):

        # project to encoder hidden size
        x_source = self.proj(x_source) # (B, SEQ, D_enc)
        
        # tuple with n_enc_layers tensors of (B, SEQ, D_enc)
        x_source = tuple([x_source for _ in range(self.n_enc_layers)])

        x_source = {"x_source": x_source}

        return self.encoder(input_ids.long(), attention_mask=attention_mask.long(), labels=labels.long(), output_hidden_states=False, **x_source)
    
    def generate(self, input_ids, attention_mask, x_source, **kwargs):

        # project to encoder hidden size
        x_source = self.proj(x_source)

        # tuple with n_enc_layers tensors of (B, SEQ, D_enc)
        x_source = [tuple([x_source for _ in range(self.n_enc_layers)])]

        x_source = {"x_source": x_source}

        return self.encoder.generate(input_ids=input_ids.long(), attention_mask=attention_mask.long(), max_new_tokens=self.generate_max_tokens, output_hidden_states=False, **x_source)