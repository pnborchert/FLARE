from peft import LoraModel
from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from peft.tuners.lora.aqlm import dispatch_aqlm
from peft.tuners.lora.awq import dispatch_awq
from peft.tuners.lora.gptq import dispatch_gptq
from peft.tuners.lora.tp_layer import dispatch_megatron
from layer import dispatch_default
from bnb import dispatch_bnb_8bit, dispatch_bnb_4bit

@staticmethod
def _create_new_module(lora_config, adapter_name, target, **kwargs):
    # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
    # because the first match is always used. Therefore, the default layers should be checked last.
    dispatchers = []

    # avoid eager bnb import
    if is_bnb_available():
        # from peft.tuners.lora.bnb import dispatch_bnb_8bit

        dispatchers.append(dispatch_bnb_8bit)

    if is_bnb_4bit_available():
        # from peft.tuners.lora.bnb import dispatch_bnb_4bit

        dispatchers.append(dispatch_bnb_4bit)

    dispatchers.extend([dispatch_aqlm, dispatch_awq, dispatch_gptq, dispatch_megatron, dispatch_default])

    new_module = None
    for dispatcher in dispatchers:
        new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
        if new_module is not None:  # first match wins
            break

    if new_module is None:
        # no module could be matched
        raise ValueError(
            f"Target module {target} is not supported. Currently, only the following modules are supported: "
            "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
        )

    return new_module

# overwrite LoraModel dispatch
LoraModel._create_new_module = _create_new_module