import torch
from .base import CLS, QA, QADecoder, Decoder
from layer import LinearFuse

class FusionCLS(CLS):
    def __init__(self, encoder, args):
        super().__init__(encoder, args)
        self.inference_mode = False

    def set_inference_mode(self, enabled=True):
        self.inference_mode = enabled
        for module in self.encoder.modules():
            if isinstance(module, LinearFuse):
                module.set_inference_mode(enabled=enabled)
    
    def forward(self, input_ids_source, attention_mask_source, input_ids_target, attention_mask_target, labels, **kwargs):

        # get hidden states for source language
        with torch.no_grad():
            self.encoder.disable_adapters()
            out = self.encoder(input_ids_source.long(), attention_mask=attention_mask_source.long())

            x_source = {}
            if "hidden_states" in out:
                x_source["x_source"] = out["hidden_states"]
            if "encoder_hidden_states" in out:
                x_source["x_source_encoder"] = out["encoder_hidden_states"]
            if "decoder_hidden_states" in out:
                x_source["x_source_decoder"] = out["decoder_hidden_states"]
        
        # get hidden states for target language
        self.encoder.enable_adapters()
        out_target = self.encoder(input_ids_target.long(), attention_mask=attention_mask_target.long(), labels=labels, output_hidden_states=False, **x_source)

        return out_target

class FusionDecoder(Decoder):
    def __init__(self, encoder, args):
        super().__init__(encoder, args)
        self.plm = args["plm"]
    
    def forward(self, input_ids_source, attention_mask_source, input_ids_target, attention_mask_target, labels, **kwargs):

        # get hidden states for source language
        with torch.no_grad():
            self.encoder.disable_adapters()

            out = self.encoder(input_ids=input_ids_source.long(), attention_mask=attention_mask_source.long())

            x_source = {}
            if "hidden_states" in out:
                x_source["x_source"] = out["hidden_states"]
        
        # get hidden states for target language
        self.encoder.enable_adapters()
        out_target = self.encoder(input_ids=input_ids_target.long(), attention_mask=attention_mask_target.long(), labels=labels.long(), output_hidden_states=False, **x_source)

        return out_target

    def generate(self, input_ids_source, attention_mask_source, input_ids_target, attention_mask_target, **kwargs):

        # get hidden states for source language
        with torch.no_grad():
            self.encoder.disable_adapters()

            out = self.encoder.generate(input_ids=input_ids_source.long(), attention_mask=attention_mask_source.long(), max_new_tokens=self.generate_max_tokens, output_hidden_states=True, return_dict_in_generate=True)

            x_source = {}
            if "hidden_states" in out:
                x_source["x_source"] = out["hidden_states"]

        # get hidden states for target language
        self.encoder.enable_adapters()
        out_target = self.encoder.generate(input_ids=input_ids_target.long(), attention_mask=attention_mask_target.long(), max_new_tokens=self.generate_max_tokens, output_hidden_states=False, **x_source)

        return out_target
    
class FusionQA(QA):
    def __init__(self, encoder, args):
        super().__init__(encoder, args)
        self.inference_mode = False

    def set_inference_mode(self, enabled=True):
        self.inference_mode = enabled
        for module in self.encoder.modules():
            if isinstance(module, LinearFuse):
                module.set_inference_mode(enabled=enabled)
    
    def forward(self, input_ids_source, attention_mask_source, input_ids_target, attention_mask_target, start_positions, end_positions, **kwargs):

        # get hidden states for source language
        with torch.no_grad():
            self.encoder.disable_adapters()
            out = self.encoder(input_ids=input_ids_source.long(), attention_mask=attention_mask_source.long())

            x_source = {}
            if "hidden_states" in out:
                x_source["x_source"] = out["hidden_states"]
            if "encoder_hidden_states" in out:
                x_source["x_source_encoder"] = out["encoder_hidden_states"]
            if "decoder_hidden_states" in out:
                x_source["x_source_decoder"] = out["decoder_hidden_states"]
        
        # get hidden states for target language
        self.encoder.enable_adapters()
        out_target = self.encoder(input_ids_target.long(), attention_mask=attention_mask_target.long(), start_positions=start_positions, end_positions=end_positions, output_hidden_states=False, **x_source)

        return out_target

class FusionQADecoder(QADecoder):
    def __init__(self, encoder, args):
        super().__init__(encoder, args)
    
    def forward(self, input_ids_source, attention_mask_source, input_ids_target, attention_mask_target, labels, **kwargs):

        # get hidden states for source language
        with torch.no_grad():
            self.encoder.disable_adapters()

            out = self.encoder(input_ids=input_ids_source.long(), attention_mask=attention_mask_source.long())

            x_source = {}
            if "hidden_states" in out:
                x_source["x_source"] = out["hidden_states"]
        
        # get hidden states for target language
        self.encoder.enable_adapters()
        out_target = self.encoder(input_ids=input_ids_target.long(), attention_mask=attention_mask_target.long(), labels=labels.long(), output_hidden_states=False, **x_source)

        return out_target

    def generate(self, input_ids_source, attention_mask_source, input_ids_target, attention_mask_target, **kwargs):

        # get hidden states for source language
        with torch.no_grad():
            self.encoder.disable_adapters()

            out = self.encoder.generate(input_ids=input_ids_source.long(), attention_mask=attention_mask_source.long(), max_new_tokens=self.generate_max_tokens, output_hidden_states=True, return_dict_in_generate=True)

            x_source = {}
            if "hidden_states" in out:
                x_source["x_source"] = out["hidden_states"]

        # get hidden states for target language
        self.encoder.enable_adapters()
        out_target = self.encoder.generate(input_ids=input_ids_target.long(), attention_mask=attention_mask_target.long(), max_new_tokens=self.generate_max_tokens, output_hidden_states=False, **x_source)

        return out_target