import torch, argparse, wandb, os, json, glob, safetensors, tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging
import pandas as pd
import numpy as np

MARKER = '"'

MODELS = {
    "nllb-600m": "facebook/nllb-200-distilled-600M",
    "nllb-3.3b": "facebook/nllb-200-3.3B",
}

TEXTFIELDS = {
    "xnli": ["premise", "hypothesis"],
    "xsquad": ["context", "question"],
    "tydiqa": ["context", "question"],
    "nusax": ["text"],
    "amnli": ["premise", "hypothesis"],
}

LANG = {
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "aym": "ayr_Latn",
    "ace": "ace_Latn",
    "ar": "arb_Arab",
    "ban": "ban_Latn",
    "ben": "ben_Beng",
    "bbc": "ind_Latn",
    "bjn": "bjn_Latn",
    "bug": "bug_Latn",
    "bg": "bul_Cyrl",
    "bzd": "spn_Latn",
    "cni": "spa_Latn",
    "el": "ell_Grek",
    "es": "spa_Latn",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "gn": "grn_Latn",
    "hch": "spa_Latn",
    "hi": "hin_Deva",
    "ind": "ind_Latn",
    "jav": "jav_Latn",
    "ko": "kor_Hang",
    "mad": "ind_Latn",
    "min": "min_Latn",
    "nah": "spa_Latn",
    "nij": "ind_Latn",
    "oto": "spa_Latn",
    "quy": "quy_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sun": "sun_Latn",
    "shp": "spa_Latn",
    "tar": "spa_Latn",
    "tel": "tel_Telu",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "ur": "urd_Arab",
    "vi": "vie_Latn",
    "sw": "swh_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
}

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--source_lang", type=str, required=True)
parser.add_argument("--target_lang", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument("--output_dir", type=str, default="data")
parser.add_argument("--save_format", type=str, default="parquet")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--path_train", type=str)
parser.add_argument("--path_val", type=str)
parser.add_argument("--path_test", type=str)

args = vars(parser.parse_args())

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, args, data, field, tokenizer):

        if field is not None:
            self.data = np.array([i[field] for i in data])
        else:
            self.data = np.array(data)
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.tokenizer(self.data[idx], return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)

        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)

        return inputs

def process_xnli(data):
    # to List[Dict[lang:str, label:int, premise:str, hypothesis:str]]

    data_out = []
    languages = list(data["premise"][0].keys())

    for i in range(len(data["label"])):
        for lang in languages:
            h_idx = data["hypothesis"][i]["language"].index(lang)
            data_out.append({
                "language": lang,
                "label": data["label"][i],
                "premise": data["premise"][i][lang],
                "hypothesis": data["hypothesis"][i]["translation"][h_idx]
            })
    
    return data_out

def process_qa(data):
    # if answers contain multiple answers, we only take the first one
    for i in range(len(data["answers"])):
        if len(data["answers"][i]["text"]) > 1:
            data["answers"][i]["text"] = [data["answers"][i]["text"][0]]
            data["answers"][i]["answer_start"] = [data["answers"][i]["answer_start"][0]]

        # remove all markers from context
        len_before = len(data["context"][i])
        data["context"][i] = data["context"][i].replace(MARKER, "")
        offset = len_before - len(data["context"][i])
        
        # surround answer text in context with markers
        start, end  = data["answers"][i]["answer_start"][0]-offset, data["answers"][i]["answer_start"][0] + len(data["answers"][i]["text"][0]) - offset
        data["context"][i] = data["context"][i][:start] + MARKER + data["context"][i][start:end] + MARKER + data["context"][i][end:]

    return data

def postprocess_qa(data):
    # find answer position in translated data
    # add "answers_translated" to data
    answers_translated = []

    for i in range(len(data["answers"])):
        # answer position in translated data
        start, end = data["context_translated"][i].find(MARKER), data["context_translated"][i].rfind(MARKER)
        context_translated = data["context_translated"][i].replace(MARKER, "")
        answer_text_translated = context_translated[start:end-1]
        answers_translated.append({"text":[answer_text_translated], "answer_start":[start]})
    
    data["answers_translated"] = answers_translated

    return data

def load_data(args):

    # specify default and alternatively overwrite with path_train (is not None) etc.

    if args["task"] == "xnli":
        dataset = load_dataset("xnli", args["source_lang"])
        train = dataset["train"].to_dict()
        val = dataset["validation"].to_dict()
        test = dataset["test"].to_dict()
    
    elif args["task"] == "amnli":

        if args["source_lang"] == "en":
            # load xnli train
            dataset = load_dataset("xnli", args["source_lang"])
            train = dataset["train"].to_dict()
            return {"train": train}
        
        else:
            dataset = load_dataset("nala-cub/americas_nli", args["source_lang"])
            train = dataset["validation"].to_dict()
            test = dataset["test"].to_dict()
            return {"train": train, "test": test}
    
    elif args["task"] == "tydiqa":

        train = pd.read_parquet(args["path_train"])
        train = train[train["lang"] == args["source_lang"]].reset_index(drop=True).to_dict(orient="list")
        train = process_qa(train)

        test = pd.read_parquet(args["path_test"])
        test = test[test["lang"] == args["source_lang"]].reset_index(drop=True).to_dict(orient="list")
        test = process_qa(test)

        return {"train": train, "test": test}
    
    elif args["task"] == "nusax":

        if args["source_lang"] == "en":
            dataset = load_dataset('indonlp/NusaX-senti', "eng")
        else:
            dataset = load_dataset('indonlp/NusaX-senti', args["source_lang"])

        train = dataset["train"].to_dict()
        val = dataset["validation"].to_dict()
        test = dataset["test"].to_dict()

        if args["source_lang"] == "eng":
            # eng -> en
            for split in [train, val, test]:
                split["lang"] = ["en" for _ in split["text"]]

    else:
        NotImplementedError

    return {"train": train, "val": val, "test": test}

def get_dataloader(args, data, tokenizer, field=None):
    
    dataset = TranslationDataset(args=args, data=data, field=field,tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args["batch_size"], shuffle=False)

    return dataloader

def load_model(args):
    
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args["model"]], src_lang=LANG[args["source_lang"]], tgt_lang=LANG[args["target_lang"]])
    model = AutoModelForSeq2SeqLM.from_pretrained(MODELS[args["model"]])

    return model, tokenizer

def translate(args, model, tokenizer, dataloader, desc=""):

    dl = iter(dataloader)
    translations = []

    for batch in tqdm.tqdm(dl, desc=desc):
        batch = {k: v.to(args["device"]) for k, v in batch.items()}
        translated_tokens = model.generate(**batch, forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG[args["target_lang"]]), max_length=args["max_length"])
        t_batch = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translations.extend(t_batch)

    return translations

def save_translations(args, data, name):

    if args["task"] in ["tydiqa"]:
        data = postprocess_qa(data)

    if args["save_format"] == "parquet":

        df = pd.DataFrame(data)
        df["language"] = args["source_lang"]
        df.to_parquet(os.path.join(args["output_dir"], f"{name}.parquet"))

    else:
        NotImplementedError

def main(args):

    logger = logging.get_logger(__name__)

    args["output_dir"] = os.path.join(args["output_dir"], args["task"], f'translate-{args["target_lang"]}-{args["model"]}', )

    logger.warning(f"{'='*10} Translate {args['source_lang']} -> {args['target_lang']} {'='*10}")
    logger.warning(f"Task: {args['task']}")
    logger.warning(f"Model: {args['model']}")
    logger.warning(f"Output directory: {args['output_dir']}")

    # create output directory
    os.makedirs(args["output_dir"], exist_ok=True)

    # load model and tokenizer
    model, tokenizer = load_model(args)

    # load data
    data_dict = load_data(args)

    # to device
    model.to(args["device"])
    model.eval()
            
    # translate (per language)
    for split, data in data_dict.items():
        for field in TEXTFIELDS[args["task"]]:
            dl = get_dataloader(args=args, data=data[field], tokenizer=tokenizer)
            translations = translate(args=args, model=model, tokenizer=tokenizer, dataloader=dl, desc=f"Translating {field} in {split}")
            assert len(translations) == len(data[field])
            
            data[f"{field}_translated"] = translations
        
        save_translations(args, data, name=f"{split}_{args['source_lang']}-{args['target_lang']}")
    
if __name__ == "__main__":
    main(args)