# 🧩 Extending FLARE to New Models (e.g., Llama)

This guide walks you through adapting FLARE to a new transformer model, using Llama as an example. All FLARE-compatible models are located in the `FLARE/models` directory.

---

## ✅ Step 1: Copy the Base Model Implementation

Copy the original LLaMA implementation from HuggingFace Transformers into your FLARE repository:

```bash
cp transformers/models/llama/modeling_llama.py FLARE/models/
```

---

## ✅ Step 2: Replace the GenerationMixin

Modify the import to use FLARE's custom `GenerationMixin`:

```python
# from transformers.generation import GenerationMixin
from utils import GenerationMixin
```

---

## ✅ Step 3: Enable `x_source` Support

To enable fusion, we need to propagate `x_source` (the hidden states from the source language model) through the target language model. This involves updating several components to accept and forward `x_source` using `**kwargs`.

### 🔧 Update `LlamaAttention`

Modify the attention module to pass `x_source` into the LoRA adapters through `q_proj` and `v_proj`.

```python
class LlamaAttention(nn.Module):

    ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        ...
        **kwargs, # <- Add this
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        ...
            x_source = kwargs.get("x_source", None) # <- Add this

            if x_source is not None:
                query_states = self.q_proj(hidden_states, x_source=x_source) # <- Add this
                value_states = self.v_proj(hidden_states, x_source=x_source) # <- Add this
            else:
                query_states = self.q_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            key_states = self.k_proj(hidden_states)

        ...
```

---

### 🔧 Update `LlamaDecoderLayer`

Ensure `**kwargs` are accepted and passed to the attention layer.

```python
class LlamaDecoderLayer(nn.Module):
    
    ...

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        ...
        **kwargs, # <- Add this
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        ...

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs, # <- Add this
        )
        ...
```

---

### 🔧 Update `LlamaModel`

Propagate `x_source[i]` to the corresponding decoder layers.

```python
class LlamaModel(LlamaPreTrainedModel):

    ...
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        ...
        **kwargs, # <- Add this
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        ...
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    x_source = kwargs.get("x_source")[i] if kwargs.get("x_source") is not None else None, # <- Add this
                )
            ...
```

---

### 🔧 Update `LlamaForCausalLM`

Make two changes here:

1. Inherit `GenerationMixin` first.
2. Accept `x_source` and pass it to the model.

```python
class LlamaForCausalLM(GenerationMixin,LlamaPreTrainedModel): # <- GenerationMixin first

    ...

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        ...,
        x_source: Optional[List[torch.FloatTensor]] = None, # <- Add this
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        ...

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            x_source = x_source, # <- Add this
        )
        ...
```

---

## 🧪 Final Notes

- Register the model in `FLARE/models/__init__.py`:

```python
from .modeling_llama import LlamaForCausalLM
```

- Add loading logic to `FLARE/run_task.py` and `FLARE/run.py` as needed.
