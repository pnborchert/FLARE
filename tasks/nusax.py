import torch, os, json, wandb, re
from datasets import load_dataset
import pandas as pd
import numpy as np
from evaluate import load
from sklearn.metrics import accuracy_score, f1_score

# constants
TYPE = "classification"
NUM_LABELS = 3
BEST_METRIC = "eval_f1"
TOKENIZER = None
VERBALIZER = {0: "negative", 1: "neutral", 2: "positive"}
VERBALIZER_R = {v: k for k, v in VERBALIZER.items()}

def get_label_names(args):
    return ["labels"]

class SADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
    
    def __len__(self):
        return len(self.data["label"])
    
    def __getitem__(self, idx):
        text = self.data["text"][idx]
        label = self.data["label"][idx]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label),
        }

class CLM_SADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, args, is_testset=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
        self.is_testset = is_testset

        self.verbalizer = args["verbalizer"]
        self.data["label"] = [self.verbalizer[label] for label in self.data["label"]]
    
    def __len__(self):
        return len(self.data["label"])
    
    def __getitem__(self, idx):
        if self.is_testset:
            return self.__getitem_test__(idx)
        else:
            return self.__getitem_train__(idx)
    
    def __getitem_test__(self, idx):
        text = self.data["text"][idx]
        label = self.data["label"][idx]

        text = f"{text}\nSentiment: "

        text = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(label, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if text["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            text["input_ids"] = text["input_ids"][:, :self.max_length-answer["input_ids"].shape[1]]
            text["attention_mask"] = text["attention_mask"][:, :self.max_length-answer["input_ids"].shape[1]]

        # train on context + answer
        labels = torch.cat([text["input_ids"], answer["input_ids"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.max_length - text["input_ids"].shape[1]

        if pad_length > 0:
            text["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), text["input_ids"]], dim=-1)
            text["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), text["attention_mask"]], dim=-1)

        pad_length = self.max_length - labels.shape[1]

        if pad_length > 0:
            labels = torch.cat([torch.full((1, pad_length), -100, dtype=torch.long), labels], dim=-1)

        return {
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
    
    def __getitem_train__(self, idx):
        text = self.data["text"][idx]
        label = self.data["label"][idx]

        text = f"{text}\nSentiment: "

        text = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(label, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if text["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            text["input_ids"] = text["input_ids"][:, :self.max_length-answer["input_ids"].shape[1]]
            text["attention_mask"] = text["attention_mask"][:, :self.max_length-answer["input_ids"].shape[1]]

        # train on context + answer
        labels = torch.cat([torch.full_like(text["input_ids"], -100), answer["input_ids"]], dim=-1)
        text["input_ids"] = torch.cat([text["input_ids"], answer["input_ids"]], dim=-1)
        text["attention_mask"] = torch.cat([text["attention_mask"], answer["attention_mask"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.max_length - text["input_ids"].shape[1]

        if pad_length > 0:
            text["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), text["input_ids"]], dim=-1)
            text["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), text["attention_mask"]], dim=-1)
            labels = torch.cat([torch.full((1, pad_length), -100, dtype=torch.long), labels], dim=-1)
        
        return {
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

class MTFusionSADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, mt_model, mt_tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]

        self.device = args["device"]
        self.mt_tokenizer = mt_tokenizer
        self.mt_model = mt_model.to(self.device)
        self.mt_model.eval()
    
    def __len__(self):
        return len(self.data["label"])
    
    def __getitem__(self, idx):
        text = self.data["text"][idx]
        label = self.data["label"][idx]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # mt tokenization
        mt_inputs = self.mt_tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        with torch.no_grad():
            mt_inputs = {k: v.to(self.device) for k, v in mt_inputs.items()}
            out = self.mt_model(**mt_inputs)
            x_source = out.last_hidden_state

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "x_source": x_source.squeeze(0),
            "labels": torch.tensor(label),
        }

class CLM_MTFusionSADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, mt_model, mt_tokenizer, args, is_testset=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
        self.is_testset = is_testset
        self.verbalizer = args["verbalizer"]
        self.data["label"] = [self.verbalizer[label] for label in self.data["label"]]

        self.device = args["device"]
        self.mt_tokenizer = mt_tokenizer
        self.mt_model = mt_model.to(self.device)
        self.mt_model.eval()
    
    def __len__(self):
        return len(self.data["label"])
    
    def __getitem__(self, idx):
        if self.is_testset:
            return self.__getitem_test__(idx)
        else:
            return self.__getitem_train__(idx)
        
    def __getitem_test__(self, idx):
        text = self.data["text"][idx]
        label = self.data["label"][idx]

        text = f"{text}\nSentiment: "

        mt_inputs = self.mt_tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        with torch.no_grad():
            mt_inputs = {k: v.to(self.device) for k, v in mt_inputs.items()}
            out = self.mt_model(**mt_inputs)
            x_source = out.last_hidden_state

        text = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(label, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if text["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            text["input_ids"] = text["input_ids"][:, :self.max_length-answer["input_ids"].shape[1]]
            text["attention_mask"] = text["attention_mask"][:, :self.max_length-answer["input_ids"].shape[1]]

        # train on context + answer
        labels = torch.cat([text["input_ids"], answer["input_ids"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.max_length - text["input_ids"].shape[1]

        if pad_length > 0:
            text["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), text["input_ids"]], dim=-1)
            text["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), text["attention_mask"]], dim=-1)

        pad_length = self.max_length - labels.shape[1]

        if pad_length > 0:
            labels = torch.cat([torch.full((1, pad_length), -100, dtype=torch.long), labels], dim=-1)

        return {
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "x_source": x_source.squeeze(0),
            "labels": labels.squeeze(0),
        }
        
    def __getitem_train__(self, idx):
        text = self.data["text"][idx]
        label = self.data["label"][idx]

        text = f"{text}\nSentiment: "

        mt_inputs = self.mt_tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        with torch.no_grad():
            mt_inputs = {k: v.to(self.device) for k, v in mt_inputs.items()}
            out = self.mt_model(**mt_inputs)
            x_source = out.last_hidden_state

        text = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(label, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if text["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            text["input_ids"] = text["input_ids"][:, :self.max_length-answer["input_ids"].shape[1]]
            text["attention_mask"] = text["attention_mask"][:, :self.max_length-answer["input_ids"].shape[1]]

        # train on context + answer
        labels = torch.cat([torch.full_like(text["input_ids"], -100), answer["input_ids"]], dim=-1)
        text["input_ids"] = torch.cat([text["input_ids"], answer["input_ids"]], dim=-1)
        text["attention_mask"] = torch.cat([text["attention_mask"], answer["attention_mask"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.max_length - text["input_ids"].shape[1]

        if pad_length > 0:
            text["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), text["input_ids"]], dim=-1)
            text["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), text["attention_mask"]], dim=-1)
            labels = torch.cat([torch.full((1, pad_length), -100, dtype=torch.long), labels], dim=-1)

        return {
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "x_source": x_source.squeeze(0),
            "labels": labels.squeeze(0),
        }

class SAFusionDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, data_target, tokenizer, args):
        self.data_source = data_source
        self.data_target = data_target
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
    
    def __len__(self):
        return len(self.data_target["label"])
    
    def __getitem__(self, idx):
        text_source = self.data_source["text"][idx]
        text_target = self.data_target["text"][idx]
        label = self.data_target["label"][idx]

        inputs_source = self.tokenizer(
            text_source,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs_target = self.tokenizer(
            text_target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids_source": inputs_source["input_ids"].squeeze(0),
            "attention_mask_source": inputs_source["attention_mask"].squeeze(0),
            "input_ids_target": inputs_target["input_ids"].squeeze(0),
            "attention_mask_target": inputs_target["attention_mask"].squeeze(0),
            "labels": torch.tensor(label),
        }
    
class CLM_SAFusionDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, data_target, tokenizer, args, is_testset=False):
        self.data_source = data_source
        self.data_target = data_target
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
        self.is_testset = is_testset

        self.verbalizer = args["verbalizer"]
        self.data_target["label"] = [self.verbalizer[label] for label in self.data_target["label"]]
    
    def __len__(self):
        return len(self.data_target["label"])
    
    def __getitem__(self, idx):
        if self.is_testset:
            return self.__getitem_test__(idx)
        else:
            return self.__getitem_train__(idx)
    
    def __getitem_test__(self, idx):
        text_source = self.data_source["text"][idx]
        text_target = self.data_target["text"][idx]
        label = self.data_target["label"][idx]

        text_source = f"{text_source}\nSentiment: "
        text_target = f"{text_target}\nSentiment: "

        text_source = self.tokenizer(text_source, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        text_target = self.tokenizer(text_target, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(label, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if text_target["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            text_target["input_ids"] = text_target["input_ids"][:, :self.max_length-answer["input_ids"].shape[1]]
            text_target["attention_mask"] = text_target["attention_mask"][:, :self.max_length-answer["input_ids"].shape[1]]

        if text_source["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            text_source["input_ids"] = text_source["input_ids"][:, :self.max_length-answer["input_ids"].shape[1]]
            text_source["attention_mask"] = text_source["attention_mask"][:, :self.max_length-answer["input_ids"].shape[1]]

        # add target context to labels
        labels = torch.cat([text_target["input_ids"], answer["input_ids"]], dim=-1)

        # padding to max_length (left padding)
        pad_length_source = self.max_length - text_source["input_ids"].shape[1]
        pad_length_target = self.max_length - text_target["input_ids"].shape[1]
        pad_length_labels = self.max_length - labels.shape[1]

        if pad_length_source > 0:
            text_source["input_ids"] = torch.cat([torch.full((1, pad_length_source), self.tokenizer.pad_token_id, dtype=torch.long), text_source["input_ids"]], dim=-1)
            text_source["attention_mask"] = torch.cat([torch.full((1, pad_length_source), 0, dtype=torch.long), text_source["attention_mask"]], dim=-1)

        if pad_length_target > 0:
            text_target["input_ids"] = torch.cat([torch.full((1, pad_length_target), self.tokenizer.pad_token_id, dtype=torch.long), text_target["input_ids"]], dim=-1)
            text_target["attention_mask"] = torch.cat([torch.full((1, pad_length_target), 0, dtype=torch.long), text_target["attention_mask"]], dim=-1)

        if pad_length_labels > 0:
            labels = torch.cat([torch.full((1, pad_length_labels), -100, dtype=torch.long), labels], dim=-1)

        return {
            "input_ids_source": text_source["input_ids"].squeeze(0),
            "attention_mask_source": text_source["attention_mask"].squeeze(0),
            "input_ids_target": text_target["input_ids"].squeeze(0),
            "attention_mask_target": text_target["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        } 
  
    def __getitem_train__(self, idx):
        text_source = self.data_source["text"][idx]
        text_target = self.data_target["text"][idx]
        label = self.data_target["label"][idx]

        text_source = f"{text_source}\nSentiment: "
        text_target = f"{text_target}\nSentiment: "

        text_source = self.tokenizer(text_source, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        text_target = self.tokenizer(text_target, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(label, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if text_target["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            text_target["input_ids"] = text_target["input_ids"][:, :self.max_length-answer["input_ids"].shape[1]]
            text_target["attention_mask"] = text_target["attention_mask"][:, :self.max_length-answer["input_ids"].shape[1]]

        if text_source["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            text_source["input_ids"] = text_source["input_ids"][:, :self.max_length-answer["input_ids"].shape[1]]
            text_source["attention_mask"] = text_source["attention_mask"][:, :self.max_length-answer["input_ids"].shape[1]]

        # train on context + answer
        labels = torch.cat([torch.full_like(text_target["input_ids"], -100), answer["input_ids"]], dim=-1)
        text_target["input_ids"] = torch.cat([text_target["input_ids"], answer["input_ids"]], dim=-1)
        text_target["attention_mask"] = torch.cat([text_target["attention_mask"], answer["attention_mask"]], dim=-1)

        # padding to max_length (left padding)
        pad_length_source = self.max_length - text_source["input_ids"].shape[1]
        pad_length_target = self.max_length - text_target["input_ids"].shape[1]

        if pad_length_source > 0:
            text_source["input_ids"] = torch.cat([torch.full((1, pad_length_source), self.tokenizer.pad_token_id, dtype=torch.long), text_source["input_ids"]], dim=-1)
            text_source["attention_mask"] = torch.cat([torch.full((1, pad_length_source), 0, dtype=torch.long), text_source["attention_mask"]], dim=-1)

        if pad_length_target > 0:
            text_target["input_ids"] = torch.cat([torch.full((1, pad_length_target), self.tokenizer.pad_token_id, dtype=torch.long), text_target["input_ids"]], dim=-1)
            text_target["attention_mask"] = torch.cat([torch.full((1, pad_length_target), 0, dtype=torch.long), text_target["attention_mask"]], dim=-1)
            labels = torch.cat([torch.full((1, pad_length_target), -100, dtype=torch.long), labels], dim=-1)

        return {
            "input_ids_source": text_source["input_ids"].squeeze(0),
            "attention_mask_source": text_source["attention_mask"].squeeze(0),
            "input_ids_target": text_target["input_ids"].squeeze(0),
            "attention_mask_target": text_target["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

class SAInputFusionDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, data_target, tokenizer, args):
        self.data_source = data_source
        self.data_target = data_target
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
    
    def __len__(self):
        return len(self.data_target["label"])
    
    def __getitem__(self, idx):
        text_source = self.data_source["text"][idx]
        text_target = self.data_target["text"][idx]
        label = self.data_target["label"][idx]

        inputs = self.tokenizer(
            text_source + text_target,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(label),
        }

class CLM_SAInputFusionDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, data_target, tokenizer, args, is_testset=False):
        self.data_source = data_source
        self.data_target = data_target
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
        self.is_testset = is_testset

        self.verbalizer = args["verbalizer"]
        self.data_target["label"] = [self.verbalizer[label] for label in self.data_target["label"]]
    
    def __len__(self):
        return len(self.data_target["label"])
    
    def __getitem__(self, idx):
        if self.is_testset:
            return self.__getitem_test__(idx)
        else:
            return self.__getitem_train__(idx)
    
    def __getitem_test__(self, idx):
        text_source = self.data_source["text"][idx]
        text_target = self.data_target["text"][idx]
        label = self.data_target["label"][idx]

        text = f"{text_source}\n{text_target}\nSentiment: "

        text = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(label, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if text["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            text["input_ids"] = text["input_ids"][:, :self.max_length-answer["input_ids"].shape[1]]
            text["attention_mask"] = text["attention_mask"][:, :self.max_length-answer["input_ids"].shape[1]]

        # add target context to labels
        labels = torch.cat([text["input_ids"], answer["input_ids"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.max_length - text["input_ids"].shape[1]
        pad_length_labels = self.max_length - labels.shape[1]

        if pad_length > 0:
            text["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), text["input_ids"]], dim=-1)
            text["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), text["attention_mask"]], dim=-1)

        if pad_length_labels > 0:
            labels = torch.cat([torch.full((1, pad_length_labels), -100, dtype=torch.long), labels], dim=-1)

        return {
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
        
    def __getitem_train__(self, idx):
        text_source = self.data_source["text"][idx]
        text_target = self.data_target["text"][idx]
        label = self.data_target["label"][idx]

        text = f"{text_source}\n{text_target}\nSentiment: "

        text = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(label, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if text["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            text["input_ids"] = text["input_ids"][:, :self.max_length-answer["input_ids"].shape[1]]
            text["attention_mask"] = text["attention_mask"][:, :self.max_length-answer["input_ids"].shape[1]]

        # train on context + answer
        labels = torch.cat([torch.full_like(text["input_ids"], -100), answer["input_ids"]], dim=-1)
        text["input_ids"] = torch.cat([text["input_ids"], answer["input_ids"]], dim=-1)
        text["attention_mask"] = torch.cat([text["attention_mask"], answer["attention_mask"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.max_length - text["input_ids"].shape[1]

        if pad_length > 0:
            text["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), text["input_ids"]], dim=-1)
            text["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), text["attention_mask"]], dim=-1)
            labels = torch.cat([torch.full((1, pad_length), -100, dtype=torch.long), labels], dim=-1)

        return {
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

# load dataset configuration
def load_data(args, lang):

    if args["translate-test"]:
        print("Loading MT data:", args["path_mt"])
        train = pd.read_parquet(f"{args['path_mt']}/train_{args['target_lang']}-{args['source_lang']}.parquet")
        val = pd.read_parquet(f"{args['path_mt']}/val_{args['target_lang']}-{args['source_lang']}.parquet")
        test = pd.read_parquet(f"{args['path_mt']}/test_{args['target_lang']}-{args['source_lang']}.parquet")

        train = train[["text_translated", "label"]].rename(columns={"text_translated": "text"}).to_dict(orient="list")
        val = val[["text_translated", "label"]].rename(columns={"text_translated": "text"}).to_dict(orient="list")
        test
        
    if (args["translate-train"]) and (lang == args["target_lang"]):
        print("Target: Loading MT training data and HF test data:", args["path_mt"])

        train = pd.read_parquet(f"{args['path_mt']}/train_{args['source_lang']}-{args['target_lang']}.parquet")
        train = train[["text_translated", "label"]].rename(columns={"text_translated": "text"}).to_dict(orient="list")

        val = pd.read_parquet(f"{args['path_mt']}/val_{args['source_lang']}-{args['target_lang']}.parquet")
        val = val[["text_translated", "label"]].rename(columns={"text_translated": "text"}).to_dict(orient="list")

        dataset = load_dataset("indonlp/NusaX-senti", lang)
        test = dataset["test"].to_dict()
    
    elif (args["translate-train"]) and (lang == args["source_lang"]):
        assert lang == "en", "Only support en as source language"

        dataset = load_dataset("indonlp/NusaX-senti", "eng")
        train = dataset["train"].to_dict()
        val = dataset["validation"].to_dict()
        
        # load translated test data
        print("Source: Loading MT test data:", args["path_mt_test"])

        test = pd.read_parquet(f"{args['path_mt_test']}/test_{args['target_lang']}-{args['source_lang']}.parquet")
        test = test[["text_translated", "label"]].rename(columns={"text_translated": "text"}).to_dict(orient="list")
    
    else:
        print("Loading HF data:", "indonlp/NusaX-senti")
        if lang == "en":
            dataset = load_dataset("indonlp/NusaX-senti", "eng")
        else:
            dataset = load_dataset("indonlp/NusaX-senti", lang)
        train = dataset["train"].to_dict()
        val = dataset["validation"].to_dict()
        test = dataset["test"].to_dict()
    
    return train, val, test

# Todo: load correct dataset and copy eval

def get_dataloader(train, val, test, tokenizer, args, **kwargs):
    if (args["fusion_fn"] is not None):
        if args["plm"].startswith("llama") or args["plm"].startswith("gemma"):
            if args["fuse_mt"]:
                ds_train = CLM_MTFusionSADataset(data=train, tokenizer=tokenizer, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"], args=args)
                ds_val = CLM_MTFusionSADataset(data=val, tokenizer=tokenizer, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"], args=args, is_testset=True)
                ds_test = CLM_MTFusionSADataset(data=test, tokenizer=tokenizer, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"], args=args, is_testset=True)
            else:
                ds_train = CLM_SAFusionDataset(data_source=train["source"], data_target=train["target"], tokenizer=tokenizer, args=args)
                ds_val = CLM_SAFusionDataset(data_source=val["source"], data_target=val["target"], tokenizer=tokenizer, args=args, is_testset=True)
                ds_test = CLM_SAFusionDataset(data_source=test["source"], data_target=test["target"], tokenizer=tokenizer, args=args, is_testset=True)
        else:
            if args["fuse_mt"]:
                ds_train = MTFusionSADataset(data=train, tokenizer=tokenizer, args=args, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"])
                ds_val = MTFusionSADataset(data=val, tokenizer=tokenizer, args=args, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"])
                ds_test = MTFusionSADataset(data=test, tokenizer=tokenizer, args=args, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"])
            else:
                ds_train = SAFusionDataset(data_source=train["source"], data_target=train["target"], tokenizer=tokenizer, args=args)
                ds_val = SAFusionDataset(data_source=val["source"], data_target=val["target"], tokenizer=tokenizer, args=args)
                ds_test = SAFusionDataset(data_source=test["source"], data_target=test["target"], tokenizer=tokenizer, args=args)
    elif args["input_fusion"]:
        if args["plm"].startswith("llama") or args["plm"].startswith("gemma"):
            ds_train = CLM_SAInputFusionDataset(data_source=train["source"], data_target=train["target"], tokenizer=tokenizer, args=args)
            ds_val = CLM_SAInputFusionDataset(data_source=val["source"], data_target=val["target"], tokenizer=tokenizer, args=args, is_testset=True)
            ds_test = CLM_SAInputFusionDataset(data_source=test["source"], data_target=test["target"], tokenizer=tokenizer, args=args, is_testset=True)
        else:
            ds_train = SAInputFusionDataset(data_source=train["source"], data_target=train["target"], tokenizer=tokenizer, args=args)
            ds_val = SAInputFusionDataset(data_source=val["source"], data_target=val["target"], tokenizer=tokenizer, args=args)
            ds_test = SAInputFusionDataset(data_source=test["source"], data_target=test["target"], tokenizer=tokenizer, args=args)
    else:
        if args["plm"].startswith("llama") or args["plm"].startswith("gemma"):
            ds_train = CLM_SADataset(data=train, tokenizer=tokenizer, args=args)
            ds_val = CLM_SADataset(data=val, tokenizer=tokenizer, args=args, is_testset=True)
            ds_test = CLM_SADataset(data=test, tokenizer=tokenizer, args=args, is_testset=True)
        else:
            ds_train = SADataset(data=train, tokenizer=tokenizer, args=args)
            ds_val = SADataset(data=val, tokenizer=tokenizer, args=args)
            ds_test = SADataset(data=test, tokenizer=tokenizer, args=args)

    return ds_train, ds_val, ds_test

def get_compute_metrics_fn(args):
    if args["plm"].startswith("llama") or args["plm"].startswith("gemma"):
        return compute_metrics_generate
    else:
        return compute_metrics_cls
    
def compute_metrics_generate(eval_preds):
    metric = load("exact_match")
    preds, labels = eval_preds
    preds = preds.astype(int)
    labels = labels.astype(int)

    # decode
    preds = [np.where(preds[i] != -100, preds[i], TOKENIZER.pad_token_id) for i in range(len(preds))]
    labels = [np.where(labels[i] != -100, labels[i], TOKENIZER.pad_token_id) for i in range(len(labels))]

    decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    # remove generation artifacts in llama
    decoded_preds = [re.sub(r"://.*", "", x) for x in decoded_preds]
    decoded_labels = [re.sub(r"://.*", "", x) for x in decoded_labels]

    # convert labels and preds to int using the verbalizer
    preds = [VERBALIZER_R.get(pred.split()[-1], -1) for pred in decoded_preds]
    labels = [VERBALIZER_R.get(label.split()[-1], -1) for label in decoded_labels]

    for i in range(len(preds)):
        if preds[i] == -1:
            if labels[i] == 0:
                preds[i] = 1
            else:
                preds[i] = 0

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    return {"accuracy": acc, "f1": f1}

def compute_metrics_cls(eval_preds):
    logits, labels = eval_preds

    if isinstance(logits, tuple):
        logits = logits[0]

    labels = labels.reshape(-1)
    preds = np.argmax(logits, axis=-1).reshape(-1)
    preds = preds.astype(int)
    labels = labels.astype(int)
    preds = preds[:len(labels)] # remove padding
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

def evaluate(trainer, ds, tokenizer, args, prefix=None, **kwargs):
    out = trainer.predict(ds)
    metrics = dict(out.metrics)    

    if prefix is not None:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

    wandb.config.update(metrics)
    wandb.run.summary.update(metrics) 

    # save as json
    metrics.update(args)
    with open(os.path.join(args["output_dir"], args["run_name"], f"results_{prefix}.json"), "w") as f:
        json.dump(metrics, f)
    
    return metrics