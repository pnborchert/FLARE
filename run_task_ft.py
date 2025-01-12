import torch.utils
import torch, argparse, os, glob, peft, wandb, shutil
import numpy as np
from transformers import Trainer, TrainingArguments, AutoTokenizer, BitsAndBytesConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.utils import logging, ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

import tasks
from lora_model import LoraModel
from config import LoraConfig, FuseConfig
from models import XLMRobertaForSequenceClassification, XLMRobertaForQuestionAnswering, LlamaForCausalLM, Gemma2ForCausalLM, CLS, FusionCLS, QA, FusionQA, QADecoder, FusionQADecoder, Decoder, FusionDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--path_data", type=str, default="translations")
parser.add_argument("--lang", type=str, default="en")
parser.add_argument("--plm", type=str, required=True)
parser.add_argument('--seed', help='', type=int, required=True)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--max_steps', type=int, default=None)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--eval_accumulation_steps", type=int, default=50)
parser.add_argument("--generate_max_tokens", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="checkpoints")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--bf16', dest="bf16", action='store_true')
parser.add_argument("--path_mt", type=str)
parser.add_argument("--optim", type=str, default="adamw_torch")
parser.add_argument("--wandb_project", type=str, default="flare")
parser.add_argument("--eval_only", dest="eval_only", action="store_true")

# Adapter parameters
parser.add_argument("--adapter", type=str, default=None)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--target_modules", nargs="+", default=["query", "value"])

args = vars(parser.parse_args())

PLMS = {
    "xlmr": "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
    "mt5-xl": "google/mt5-xl",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "gemma2-9b": "google/gemma-2-9b",
}

CLS_MODELS = {
    "xlmr": XLMRobertaForSequenceClassification,
    "xlmr-large": XLMRobertaForSequenceClassification,
    # "mt5-xl": MT5ForSequenceClassification,
    "llama3-8b": LlamaForCausalLM,
    "llama3.1-8b": LlamaForCausalLM,
    "gemma2-9b": Gemma2ForCausalLM,
}

QA_MODELS = {
    "xlmr": XLMRobertaForQuestionAnswering,
    "xlmr-large": XLMRobertaForQuestionAnswering,
    # "mt5-xl": MT5ForQuestionAnswering,
    "llama3-8b": LlamaForCausalLM,
    "llama3.1-8b": LlamaForCausalLM,
    "gemma2-9b": Gemma2ForCausalLM,
}

TOKENIZERS = {
    "xlmr": AutoTokenizer,
    "xlmr-large": AutoTokenizer,
    "mt5-xl": AutoTokenizer,
    "llama3-8b": AutoTokenizer,
    "llama3.1-8b": AutoTokenizer,
    "gemma2-9b": AutoTokenizer,
}

TRAINERS = {
    "seq2seq": Seq2SeqTrainer,
    "cls": Trainer,
}

TRAINERARGS = {
    "seq2seq": Seq2SeqTrainingArguments,
    "cls": TrainingArguments,
}

# resolve args
task = tasks.SUPPORTED[args["task"]]
args["verbalizer"] = task.VERBALIZER
args["fusion_fn"] = None
args["source_lang"] = None
args["translate-test"] = False
args["translate-train"] = False
args["input_fusion"] = False
args["target_lang"] = args["lang"]

kwargs_loading = {}
if (args["task"] in ["tydiqa", "xnli", "nusax"]) and (args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]):
    args["trainer"] = "seq2seq"
else:
    args["trainer"] = "cls"

if (args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]) and (args["target_modules"] == ["query", "value"]):
    args["target_modules"] = ["q_proj", "v_proj"]
    args["optim"] = "adamw_bnb_8bit"
if (args["plm"] == "mt5-xl") and (args["target_modules"] == ["query", "value"]):
    args["target_modules"] = ["q", "v"]
if args["plm"] in ["gemma2-9b"]:
    kwargs_loading["torch_dtype"] = torch.float

# set seed
torch.manual_seed(args["seed"])
np.random.seed(args["seed"])

@dataclass
class CrossAttentionConfig:
    num_attention_heads: int
    hidden_size: int
    dropout: float

class FusionTrainer(TRAINERS[args["trainer"]]):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        return model.compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
        )

def load_model(args, encoder):

    if task.TYPE in ["classification"]:
        if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]:
            if args["fusion_fn"] is not None:
                model = FusionDecoder(encoder, args)
            else:
                model = Decoder(encoder, args)
        else:
            if args["fusion_fn"] is not None:
                model = FusionCLS(encoder, args)
            else:
                model = CLS(encoder, args)

    elif task.TYPE in ["qa"]:
        if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]: 
            if args["fusion_fn"] is not None:
                model = FusionQADecoder(encoder, args)
            else:
                model = QADecoder(encoder, args)
        else:
            if args["fusion_fn"] is not None:
                model = FusionQA(encoder, args)
            else:
                model = QA(encoder, args)
    return model

def main(args):

    logger = logging.get_logger("transformers")

    args["plm_name"] = PLMS[args["plm"]]
    args["run_name"] = f"{args['task']}-{args['lang']}-{args['plm']}-{args['seed']}"

    # load tokenizer
    tokenizer = TOKENIZERS[args["plm"]].from_pretrained(args["plm_name"])
    task.TOKENIZER = tokenizer

    # load quantization config
    qconfig = None

    if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]:
        qconfig = BitsAndBytesConfig(
            # load_in_8bit=True,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        args["target_modules"] = "all-linear"

    # load plm
    if task.TYPE in ["classification"]:
        encoder = CLS_MODELS[args["plm"]].from_pretrained(args["plm_name"], num_labels=task.NUM_LABELS, output_hidden_states=True, quantization_config=qconfig, attn_implementation="eager", **kwargs_loading)
    elif task.TYPE in ["qa"]:
        encoder = QA_MODELS[args["plm"]].from_pretrained(args["plm_name"], output_hidden_states=True, quantization_config=qconfig, attn_implementation="eager", **kwargs_loading)

    if args["plm"] in ["llama3-8b", "llama3.1-8b"]:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        encoder.config.pad_token_id = tokenizer.pad_token_id
        encoder.resize_token_embeddings(len(tokenizer)) 
        tokenizer.padding_side = "left"
        encoder.config.padding_side = "left"

    if qconfig is not None:
        encoder = peft.prepare_model_for_kbit_training(encoder)
    
    # load model
    model = load_model(args, encoder)

    fuse_config = None

    if args["adapter"] == "lora":
        lora_config = LoraConfig(
            r=args["lora_r"],
            lora_alpha=args["lora_alpha"],
            target_modules=args["target_modules"],
            fuse_config=fuse_config,
        )

        model.encoder = LoraModel(model.encoder, lora_config, f"{args['adapter']}")
    
    # unfreeze task heads
    if task.TYPE in ["classification"]:
        if args["plm"] in ["xlmr", "xlmr-large"]:
            for param in model.encoder.classifier.parameters():
                param.requires_grad = True
        elif args["plm"] in ["mt5-xl"]:
            for param in model.encoder.classification_head.parameters():
                param.requires_grad = True
        elif args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]: 
            for param in model.encoder.lm_head.parameters():
                param.requires_grad = True
    
    if task.TYPE in ["qa"]:
        if args["plm"] in ["xlmr", "xlmr-large", "mt5-xl"]:
            for param in model.encoder.qa_outputs.parameters():
                param.requires_grad = True
        elif args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]:
            for param in model.encoder.lm_head.parameters():
                param.requires_grad = True
    
    # trainable cross-attention parameters
    if args["fusion_fn"] == "cross-attention":
        logger.warning("Unfreezing cross-attention parameters")
        for name, param in model.named_parameters():
            if "cross_attention" in name:
                param.requires_grad = True
    
    # unfreeze token embeddings for llama
    if args["plm"] in ["llama3-8b"]:
        for param in model.encoder.model.embed_tokens.parameters():
            param.requires_grad = True
    elif args["plm"] in ["llama3.1-8b"]:
        for param in model.encoder.model.model.embed_tokens.parameters():
            param.requires_grad = True
    
    # load data
    train, val, test = task.load_data(args, args["target_lang"])
    ds_train, ds_val, ds_test = task.get_dataloader(train, val, test, tokenizer, args)

    # args nparam
    args["param_total"] = sum(p.numel() for p in model.parameters())
    args["param_train"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    args["batch_size"] = min(len(ds_train), args["batch_size"])
    if args["max_steps"] is None:
        args["max_steps"] = len(ds_train) * args["epochs"] // args["batch_size"] // args["gradient_accumulation_steps"]
    
    args["eval_steps"] = max(args["max_steps"] // 10, 1)
    args["logging_steps"] = max(args["max_steps"] // 100, 1)

    # training arguments
    training_args = TRAINERARGS[args["trainer"]](
        output_dir=os.path.join(args["output_dir"], args["run_name"]),
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args["eval_steps"],
        eval_steps=args["eval_steps"],
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["batch_size"],
        gradient_accumulation_steps=args["gradient_accumulation_steps"],
        learning_rate=args["lr"],
        lr_scheduler_type="linear", 
        warmup_ratio=0.1,
        max_steps=args["max_steps"],
        logging_dir=args["output_dir"],
        logging_first_step=True,
        logging_steps=args["logging_steps"],
        seed=args["seed"],
        load_best_model_at_end=False if args["task"] in ["tydiqa"] else True,
        metric_for_best_model=task.BEST_METRIC,
        greater_is_better=True,
        report_to="wandb",
        label_names=task.get_label_names(args),
        run_name=args["run_name"],
        bf16=args["bf16"],
        use_cpu=args["device"] == "cpu",
        eval_accumulation_steps=args["eval_accumulation_steps"],
        save_safetensors=False,
        optim=args["optim"],
    )

    if args["trainer"] == "seq2seq":
        training_args.predict_with_generate = True

    # init wandb
    wandb.init(
        project=args["wandb_project"],
        name=args["run_name"],
        config=args,
    )

    # trainer
    trainer = FusionTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=task.get_compute_metrics_fn(args),
    )

    if not args["eval_only"]:
        trainer.train()
        trainer.save_model()

    # test
    task.evaluate(trainer=trainer, ds=ds_test, tokenizer=tokenizer, args=args, prefix="test")

    wandb.finish()

if __name__ == "__main__":
    main(args)