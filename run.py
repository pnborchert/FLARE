import torch.utils
import torch, argparse, os, glob, peft, wandb
import numpy as np
from transformers import Trainer, TrainingArguments, AutoTokenizer, BitsAndBytesConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM
from transformers.utils import logging
from dataclasses import dataclass

import tasks
from lora_model import LoraModel
from config import LoraConfig, FuseConfig
from models import XLMRobertaForSequenceClassification, XLMRobertaForQuestionAnswering, LlamaForCausalLM, Gemma2ForCausalLM, CLS, FusionCLS, QA, FusionQA, QADecoder, FusionQADecoder, FuseMTCLS, FuseMTQA, FuseMTQADecoder, Decoder, FusionDecoder, FuseMTDecoder

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--path_data", type=str, default="translations")
parser.add_argument("--source_lang", type=str, default="en")
parser.add_argument("--target_lang", type=str, required=True)
parser.add_argument("--plm", type=str, required=True)
parser.add_argument('--seed', help='', type=int, required=True)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--max_steps', type=int, default=None)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--eval_accumulation_steps", type=int, default=20)
parser.add_argument("--generate_max_tokens", type=int, default=100)
parser.add_argument("--output_dir", type=str, default="checkpoints")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument('--bf16', dest="bf16", action='store_true')
parser.add_argument("--load_ckpt", type=str, default=None)
parser.add_argument("--eval_zs", dest="eval_zs", action="store_true")
parser.add_argument("--eval_only", dest="eval_only", action="store_true")
parser.add_argument("--translate-test", dest="translate-test", action="store_true")
parser.add_argument("--translate-train", dest="translate-train", action="store_true")
parser.add_argument("--path_mt", type=str)
parser.add_argument("--optim", type=str, default="adamw_torch")
parser.add_argument("--wandb_project", type=str, default="flare")

# Adapter parameters
parser.add_argument("--adapter", type=str, default=None)
parser.add_argument("--fusion_fn", type=str, default=None)
parser.add_argument("--cross_attention_heads", type=int, default=1)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--target_modules", nargs="+", default=["query", "value"])

# fusemt parameters
parser.add_argument("--mt_model", type=str, default=None)
parser.add_argument("--fuse_mt", dest="fuse_mt", action="store_true")

# input-level fusion
parser.add_argument("--input_fusion", dest="input_fusion", action="store_true")

args = vars(parser.parse_args())

PLMS = {
    "xlmr": "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
    "mt5-xl": "google/mt5-xl",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
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

MTMODELS = {
    "nllb-600m": "facebook/nllb-200-distilled-600M",
    "nllb-3.3b": "facebook/nllb-200-3.3B",
    "nllb-54b": "facebook/nllb-moe-54b",
}

LANGS = {
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "ace": "ace_Latn",
    "ar": "arb_Arab",
    "ban": "ban_Latn",
    "ben": "ben_Beng",
    "bbc": "ind_Latn",
    "bjn": "bjn_Latn",
    "bug": "bug_Latn",
    "bg": "bul_Cyrl",
    "el": "ell_Grek",
    "es": "spa_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "ind": "ind_Latn",
    "jav": "jav_Latn",
    "ko": "kor_Hang",
    "mad": "ind_Latn",
    "min": "min_Latn",
    "nij": "ind_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sun": "sun_Latn",
    "tel": "tel_Telu",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "ur": "urd_Arab",
    "vi": "vie_Latn",
    "sw": "swh_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
}

# resolve args
task = tasks.SUPPORTED[args["task"]]
args["verbalizer"] = task.VERBALIZER

if args["eval_zs"] or args["translate-test"]:
    # ensure no new adapters are added to the model
    args["adapter"] = None

kwargs_loading = {}
if (args["task"] in ["tydiqa", "xnli", "nusax"]) and (args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]):
    args["trainer"] = "seq2seq"
else:
    args["trainer"] = "cls"

if args["translate-test"]:
    args["eval_only"] = True

if args["input_fusion"]:
    args["max_length"] = int(2 * args["max_length"])
    if args["fusion_fn"] is not None:
        raise ValueError("Cannot use input-level fusion with fusion_fn.")

if args["path_mt"] is not None:
    args["path_mt_test"] = args["path_mt"].replace(f'-{args["target_lang"]}-', f'-{args["source_lang"]}-')

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
    def compute_loss(self, model, inputs, return_outputs=False,  num_items_in_batch = None):
        return model.compute_loss(
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
        )

def load_model(args, encoder):

    if task.TYPE in ["classification"]:
        if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]:
            if args["fusion_fn"] is not None:
                if args["fuse_mt"]:
                    model = FuseMTDecoder(encoder, args)
                else:
                    model = FusionDecoder(encoder, args)
            else:
                model = Decoder(encoder, args)
        else:
            if args["fusion_fn"] is not None:
                if args["fuse_mt"]:
                    model = FuseMTCLS(encoder, args)
                else:
                    model = FusionCLS(encoder, args)
            else:
                model = CLS(encoder, args)

    elif task.TYPE in ["qa"]:
        if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]:
            if args["fusion_fn"] is not None:
                if args["fuse_mt"]:
                    model = FuseMTQADecoder(encoder, args)
                else:
                    model = FusionQADecoder(encoder, args)
            else:
                model = QADecoder(encoder, args)
        else:
            if args["fusion_fn"] is not None:
                if args["fuse_mt"]:
                    model = FuseMTQA(encoder, args)
                else:
                    model = FusionQA(encoder, args)
            else:
                model = QA(encoder, args)
    return model

def main(args):

    logger = logging.get_logger("transformers")

    # load tokenizer
    args["plm_name"] = PLMS[args["plm"]]
    tokenizer = TOKENIZERS[args["plm"]].from_pretrained(args["plm_name"])
    tokenizer.truncation_side='left'
    task.TOKENIZER = tokenizer

    # load mt tokenizer and model
    mt_tokenizer = None
    mt_model = None
    if args["fuse_mt"]:
        mt_tokenizer = AutoTokenizer.from_pretrained(MTMODELS[args["mt_model"]], src_lang=LANGS[args["source_lang"]], tgt_lang=LANGS[args["target_lang"]])
        mt_model = AutoModelForSeq2SeqLM.from_pretrained(MTMODELS[args["mt_model"]]).model.encoder
        args["mt_model_dim"] = mt_model.config.d_model

    args["run_name"] = f"{args['task']}-{args['source_lang']}-{args['target_lang']}-{args['plm']}-{args['seed']}"
    args["adapter_name"] = "lora-default"

    if args["adapter"] is not None:
        args["run_name"] += f"-{args['adapter']}"
        args["run_name"] += f"-{args['lora_r']}"

    if args["fusion_fn"] is not None:
        args["run_name"] += f"_{args['fusion_fn']}"
        args["adapter_name"] = f"flare-{args['fusion_fn']}"
    
    if args["path_mt"] is not None:
        path_mt_model = args["path_mt"].rsplit("/")[-1]
        args["run_name"] += f"-{path_mt_model}"
    
    if args["translate-test"]:
        args["run_name"] += "-ttest"
    
    if args["translate-train"]:
        args["run_name"] += "-ttrain"
    
    if args["fuse_mt"]:
        args["run_name"] += "-fusemt"
        args["adapter_name"] = f"flare-fusemt"

    if args["input_fusion"]:    
        args["run_name"] += "-inputfusion"
        args["adapter_name"] = "inputfusion"

    # load quantization config
    qconfig = None

    if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]:
        qconfig = BitsAndBytesConfig(
            # load_in_8bit=True,
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
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
        encoder.use_cache = False
        encoder.gradient_checkpointing_disable()

    if args["plm"] in ["gemma2-9b"]:
        tokenizer.add_eos_token = True
        encoder.gradient_checkpointing_disable()

    if qconfig is not None:
        encoder = peft.prepare_model_for_kbit_training(encoder)

    # load model
    model = load_model(args, encoder)
    
    # load weights from checkpoint
    if args["load_ckpt"] is not None:
        
        # load fine-tuned model with lora adapter
        lora_config = LoraConfig(
            r=args["lora_r"],
            lora_alpha=args["lora_alpha"],
            target_modules=args["target_modules"],
            fuse_config=None,
        )

        model.encoder = LoraModel(model.encoder, lora_config, "lora")

        # load weights from checkpoint
        model.load_from_ckpt(args["load_ckpt"])

    # load new adapter
    if args["fusion_fn"] is not None:

        dropout = 0.0
        if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]:
            dropout = model.encoder.config.attention_dropout
        elif args["plm"] in ["mt5-xl"]:
            dropout = model.encoder.config.dropout_rate
        elif args["plm"] in ["xlmr", "xlmr-large"]:
            dropout = model.encoder.config.attention_probs_dropout_prob

        fuse_config = FuseConfig(fusion_fn=args["fusion_fn"])
        if args["fusion_fn"] == "cross-attention":
            cross_attention_config = CrossAttentionConfig(
                num_attention_heads=args["cross_attention_heads"],
                hidden_size=args["lora_r"],
                dropout=dropout,
            )
            fuse_config.cross_attention_config = cross_attention_config
    else:
        fuse_config = None

    if args["adapter"] == "lora":

        # merge lora adapter weights
        model.encoder.merge_and_unload()

        lora_config = LoraConfig(
            r=args["lora_r"],
            lora_alpha=args["lora_alpha"],
            target_modules=args["target_modules"],
            fuse_config=fuse_config,
        )

        model.encoder.add_adapter(lora_config, adapter_name=f"{args['adapter_name']}")
    
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
    
    # unfreeze proj layer for fusemt
    if args["fuse_mt"]:
        for param in model.proj.parameters():
            param.requires_grad = True
 
    # load data
    if ((args["fusion_fn"] is not None) and (not args["fuse_mt"])) or (args["input_fusion"]):
        train_source , val_source, test_source = task.load_data(args, args["source_lang"])
        train_target, val_target, test_target = task.load_data(args, args["target_lang"])

        if args["task"] in ["tydiqa"]:
            train_source , train_target = task.filter_qa_translations(train_source, train_target)
            val_source , val_target = task.filter_qa_translations(val_source, val_target)
            test_source , test_target = task.filter_qa_translations(test_source, test_target)

        train = {"source": train_source, "target": train_target}
        val = {"source": val_source, "target": val_target}
        test = {"source": test_source, "target": test_target}

        ds_train, ds_val, ds_test = task.get_dataloader(train, val, test, tokenizer, args, mt_tokenizer=mt_tokenizer, mt_model=mt_model)
    else:
        train, val, test = task.load_data(args, args["target_lang"])

        if args["task"] in ["tydiqa"]:
            train_source , val_source, test_source = task.load_data(args, args["source_lang"])

            _, train = task.filter_qa_translations(train_source, train)
            _, val = task.filter_qa_translations(val_source, val)
            _, test = task.filter_qa_translations(test_source, test)

        ds_train, ds_val, ds_test = task.get_dataloader(train, val, test, tokenizer, args, mt_tokenizer=mt_tokenizer, mt_model=mt_model)

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
        do_predict=True,
        eval_strategy="steps",
        save_strategy="steps",
        # save_total_limit=1,
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

    if args["fuse_mt"]:
        training_args.dataloader_pin_memory = False

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

    if args["eval_zs"]:
        res = task.evaluate(trainer=trainer, ds=ds_test, tokenizer=tokenizer, args=args, prefix="zs")
        logger.warning(f"Zero-Shot: {res}")

    if not args["eval_only"]:
        trainer.train()
        trainer.save_model()

    # test
    task.evaluate(trainer=trainer, ds=ds_test, tokenizer=tokenizer, args=args, prefix="test")

    wandb.finish()

if __name__ == "__main__":
    main(args)