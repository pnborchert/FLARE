import torch, os, json, wandb, re
from datasets import load_dataset
import pandas as pd
import numpy as np
from evaluate import load
from tqdm import tqdm

# constants
TYPE = "qa"
BEST_METRIC = "eval_exact_match"
TOKENIZER = None
VERBALIZER = None

def get_label_names(args):
    if args["trainer"] == "seq2seq":
        return ["labels"]
    else:
        return ["start_positions", "end_positions"]

# datasets
class QADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
    
    def __len__(self):
        return len(self.data["question"])
    
    def __getitem__(self, idx):
        question = self.data["question"][idx]
        context = self.data["context"][idx]
        answer = self.data["answers"][idx]
        start_character = answer["answer_start"][0]
        end_character = start_character + len(answer["text"][0])
        
        # Tokenize without special tokens
        encodings = self.tokenizer(
            question,
            context,
            add_special_tokens=False,
            max_length=self.max_length-1,
            truncation="only_second",
            padding=False,
            # padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # Extract offsets and input IDs
        offsets = encodings["offset_mapping"].squeeze().tolist()
        
        # Determine the token indices for the context
        sequence_ids = encodings.sequence_ids()
        context_start = sequence_ids.index(1)  # The first token of the context
        context_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1 # The last token of the context

        start_position = None
        end_position = None

        for i in range(context_start, context_end + 1):
            if offsets[i][0] <= start_character < offsets[i][1]:
                start_position = i
            if offsets[i][0] < end_character <= offsets[i][1]:
                end_position = i
                break

        if start_position is None or end_position is None or start_position >= self.max_length or end_position >= self.max_length:
            start_position = 0
            end_position = 0

        encodings["start_positions"] = torch.tensor(start_position, dtype=torch.long)
        encodings["end_positions"] = torch.tensor(end_position, dtype=torch.long)
        encodings.pop("offset_mapping")

        # add eos token after the context
        encodings["input_ids"] = torch.cat([encodings["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        encodings["attention_mask"] = torch.cat([encodings["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # padding to max_length 
        pad_length = self.max_length - encodings["input_ids"].shape[1]

        if pad_length > 0:
            if self.tokenizer.padding_side == "right":
                encodings["input_ids"] = torch.cat([encodings["input_ids"], torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long)], dim=-1)
                encodings["attention_mask"] = torch.cat([encodings["attention_mask"], torch.full((1, pad_length), 0, dtype=torch.long)], dim=-1)
            else:
                encodings["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), encodings["input_ids"]], dim=-1)
                encodings["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), encodings["attention_mask"]], dim=-1)

        for k, v in encodings.items():
            encodings[k] = v.squeeze(0)

        return encodings

class QAFusionDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, data_target, tokenizer, args):
        self.data_source = data_source
        self.data_target = data_target
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]

    def __len__(self):
        return len(self.data_source["question"])
    
    def tokenize(self, question, context, answer):
        start_character = answer["answer_start"][0]
        end_character = start_character + len(answer["text"][0])

        # Tokenize without special tokens
        encodings = self.tokenizer(
            question,
            context,
            add_special_tokens=False,
            max_length=self.max_length-1,
            truncation="only_second",
            # padding="max_length",
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # Extract offsets and input IDs
        offsets = encodings["offset_mapping"].squeeze().tolist()

        # Determine the token indices for the context
        sequence_ids = encodings.sequence_ids()
        context_start = sequence_ids.index(1)  # The first token of the context
        context_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1  # The last token of the context

        start_position = None
        end_position = None

        for i in range(context_start, context_end + 1):
            if offsets[i][0] <= start_character < offsets[i][1]:
                start_position = i
            if offsets[i][0] < end_character <= offsets[i][1]:
                end_position = i
                break

        if start_position is None or end_position is None or start_position >= self.max_length or end_position >= self.max_length:
            start_position = 0
            end_position = 0

        encodings["start_positions"] = torch.tensor(start_position, dtype=torch.long)
        encodings["end_positions"] = torch.tensor(end_position, dtype=torch.long)
        encodings.pop("offset_mapping")

        # add eos token after the context
        encodings["input_ids"] = torch.cat([encodings["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        encodings["attention_mask"] = torch.cat([encodings["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # padding to max_length
        pad_length = self.max_length - encodings["input_ids"].shape[1]

        if pad_length > 0:
            if self.tokenizer.padding_side == "right":
                encodings["input_ids"] = torch.cat([encodings["input_ids"], torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long)], dim=-1)
                encodings["attention_mask"] = torch.cat([encodings["attention_mask"], torch.full((1, pad_length), 0, dtype=torch.long)], dim=-1)
            else:
                encodings["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), encodings["input_ids"]], dim=-1)
                encodings["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), encodings["attention_mask"]], dim=-1)

        for k, v in encodings.items():
            encodings[k] = v.squeeze(0)   

        return encodings
        

    def __getitem__(self, idx):
        inputs_source = self.tokenize(
            question=self.data_source["question"][idx],
            context=self.data_source["context"][idx],
            answer=self.data_source["answers"][idx],
            )
        
        inputs_target = self.tokenize(
            question=self.data_target["question"][idx],
            context=self.data_target["context"][idx],
            answer=self.data_target["answers"][idx],
            )

        return {
            "input_ids_source": inputs_source["input_ids"],
            "attention_mask_source": inputs_source["attention_mask"],
            "start_positions": inputs_target["start_positions"],
            "end_positions": inputs_target["end_positions"],
            "input_ids_target": inputs_target["input_ids"],
            "attention_mask_target": inputs_target["attention_mask"],
        }
    
class QAInputFusionDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, data_target, tokenizer, args):
        self.data_source = data_source
        self.data_target = data_target
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]

    def __len__(self):
        return len(self.data_source["question"])
    
    def tokenize(self, question, context, answer):
        start_character = answer["answer_start"][0]
        end_character = start_character + len(answer["text"][0])

        # Tokenize without special tokens
        encodings = self.tokenizer(
            question,
            context,
            add_special_tokens=False,
            max_length=(self.max_length // 2)-1, # divide by 2 to fit both source and target
            truncation="only_second",
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # Extract offsets and input IDs
        offsets = encodings["offset_mapping"].squeeze().tolist()

        # Determine the token indices for the context
        sequence_ids = encodings.sequence_ids()
        context_start = sequence_ids.index(1)  
        context_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1  

        start_position = None
        end_position = None

        for i in range(context_start, context_end + 1):
            if offsets[i][0] <= start_character < offsets[i][1]:
                start_position = i
            if offsets[i][0] < end_character <= offsets[i][1]:
                end_position = i
                break

        if start_position is None or end_position is None or start_position >= self.max_length or end_position >= self.max_length:
            start_position = 0
            end_position = 0

        encodings["start_positions"] = torch.tensor(start_position, dtype=torch.long)
        encodings["end_positions"] = torch.tensor(end_position, dtype=torch.long)
        encodings.pop("offset_mapping")

        for k, v in encodings.items():
            encodings[k] = v.squeeze(0)   

        return encodings
        
    def __getitem__(self, idx):
        inputs_source = self.tokenize(
            question=self.data_source["question"][idx],
            context=self.data_source["context"][idx],
            answer=self.data_source["answers"][idx],
            )
        
        inputs_target = self.tokenize(
            question=self.data_target["question"][idx],
            context=self.data_target["context"][idx],
            answer=self.data_target["answers"][idx],
            )
        
        inputs = {}
        inputs["input_ids"] = torch.cat([inputs_source["input_ids"], inputs_target["input_ids"], torch.tensor([self.tokenizer.eos_token_id])], dim=-1)
        inputs["attention_mask"] = torch.cat([inputs_source["attention_mask"], inputs_target["attention_mask"], torch.tensor([1])], dim=-1)

        # add source length to start_positions and end_positions
        inputs["start_positions"] = inputs_target["start_positions"] + inputs_source["input_ids"].shape[0]
        inputs["end_positions"] = inputs_target["end_positions"] + inputs_source["input_ids"].shape[0]

        # padding to max_length right padding
        pad_length = self.max_length - inputs["input_ids"].shape[0]

        if pad_length > 0:
            inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.full(( pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)], dim=-1)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.full(( pad_length,), 0, dtype=torch.long)], dim=-1)

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "start_positions": inputs["start_positions"],
            "end_positions": inputs["end_positions"],
        }

# datasets
class MTFusionQADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, mt_model, mt_tokenizer, args):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]

        self.device = args["device"]
        self.mt_tokenizer = mt_tokenizer
        self.mt_model = mt_model.to(self.device)
        self.mt_model.eval()
    
    def __len__(self):
        return len(self.data["question"])
    
    def __getitem__(self, idx):
        question = self.data["question"][idx]
        context = self.data["context"][idx]
        answer = self.data["answers"][idx]
        start_character = answer["answer_start"][0]
        end_character = start_character + len(answer["text"][0])

        # mt tokenization
        mt_inputs = self.mt_tokenizer(question + context, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        with torch.no_grad():
            mt_inputs = {k: v.to(self.device) for k, v in mt_inputs.items()}
            out = self.mt_model(**mt_inputs)
            x_source = out.last_hidden_state
        
        # Tokenize without special tokens
        encodings = self.tokenizer(
            question,
            context,
            add_special_tokens=False,
            max_length=self.max_length-1,
            truncation="only_second",
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        # Extract offsets and input IDs
        offsets = encodings["offset_mapping"].squeeze().tolist()
        
        # Determine the token indices for the context
        sequence_ids = encodings.sequence_ids()
        context_start = sequence_ids.index(1)  # The first token of the context
        context_end = len(sequence_ids) - sequence_ids[::-1].index(1) - 1 # The last token of the context

        start_position = None
        end_position = None

        for i in range(context_start, context_end + 1):
            if offsets[i][0] <= start_character < offsets[i][1]:
                start_position = i
            if offsets[i][0] < end_character <= offsets[i][1]:
                end_position = i
                break

        if start_position is None or end_position is None or start_position >= self.max_length or end_position >= self.max_length:
            start_position = 0
            end_position = 0

        encodings["start_positions"] = torch.tensor(start_position, dtype=torch.long)
        encodings["end_positions"] = torch.tensor(end_position, dtype=torch.long)
        encodings.pop("offset_mapping")

        encodings["x_source"] = x_source

        # add eos token after the context
        encodings["input_ids"] = torch.cat([encodings["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        encodings["attention_mask"] = torch.cat([encodings["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # padding to max_length
        pad_length = self.max_length - encodings["input_ids"].shape[1]

        if pad_length > 0:
            if self.tokenizer.padding_side == "right":
                encodings["input_ids"] = torch.cat([encodings["input_ids"], torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long)], dim=-1)
                encodings["attention_mask"] = torch.cat([encodings["attention_mask"], torch.full((1, pad_length), 0, dtype=torch.long)], dim=-1)
            else:
                encodings["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), encodings["input_ids"]], dim=-1)
                encodings["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), encodings["attention_mask"]], dim=-1)

        for k, v in encodings.items():
            encodings[k] = v.squeeze(0)

        return encodings

class CLM_QADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, args, is_testset=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
        self.is_testset = is_testset

    def __len__(self):
        return len(self.data["question"])
    
    def __getitem__(self, idx):
        if self.is_testset:
            return self.__getitem_test__(idx)
        else:
            return self.__getitem_train__(idx)
    
    def __getitem_test__(self, idx):
        question = self.data["question"][idx]
        context = self.data["context"][idx]
        answer = self.data["answers"][idx]["text"][0]

        text = f"Context: {context}\nQuestion: {question}\nAnswer: "

        context = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(answer, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if context["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            context["input_ids"] = context["input_ids"][:, :self.max_length - answer["input_ids"].shape[1]]
            context["attention_mask"] = context["attention_mask"][:, :self.max_length - answer["input_ids"].shape[1]]
        
        # concat

        # train on context + answer
        labels = torch.cat([context["input_ids"], answer["input_ids"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.max_length - context["input_ids"].shape[1]

        if pad_length > 0:
            context["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), context["input_ids"]], dim=-1)
            context["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), context["attention_mask"]], dim=-1)
        
        pad_length = self.max_length - labels.shape[1]

        if pad_length > 0:
            labels = torch.cat([torch.full((1, pad_length), -100, dtype=torch.long), labels], dim=-1)
        
        return {
            "input_ids": context["input_ids"].squeeze(0),
            "attention_mask": context["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
    
    def __getitem_train__(self, idx):
        question = self.data["question"][idx]
        context = self.data["context"][idx]
        answer = self.data["answers"][idx]["text"][0] 
        
        text = f"Context: {context}\nQuestion: {question}\nAnswer: "

        context = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(answer, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if context["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            context["input_ids"] = context["input_ids"][:, :self.max_length - answer["input_ids"].shape[1]]
            context["attention_mask"] = context["attention_mask"][:, :self.max_length - answer["input_ids"].shape[1]]
        
        # train on context + answer
        labels = torch.cat([torch.full_like(context["input_ids"], -100), answer["input_ids"]], dim=-1)
        context["input_ids"] = torch.cat([context["input_ids"], answer["input_ids"]], dim=-1)
        context["attention_mask"] = torch.cat([context["attention_mask"], answer["attention_mask"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.max_length - context["input_ids"].shape[1]

        if pad_length > 0:
            context["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), context["input_ids"]], dim=-1)
            context["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), context["attention_mask"]], dim=-1)
            labels = torch.cat([ torch.full((1, pad_length), -100, dtype=torch.long), labels], dim=-1)
        
        return {
            "input_ids": context["input_ids"].squeeze(0),
            "attention_mask": context["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

class MTFusionCLM_QADataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, mt_model, mt_tokenizer, args, is_testset=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = args["max_length"]
        self.is_testset = is_testset

        self.device = args["device"]
        self.mt_tokenizer = mt_tokenizer
        self.mt_model = mt_model.to(self.device)
        self.mt_model.eval()

    def __len__(self):
        return len(self.data["question"])
    
    def __getitem__(self, idx):
        if self.is_testset:
            return self.__getitem_test__(idx)
        else:
            return self.__getitem_train__(idx)
    
    def __getitem_test__(self, idx):
        question = self.data["question"][idx]
        context = self.data["context"][idx]
        answer = self.data["answers"][idx]["text"][0]

        text = f"Context: {context}\nQuestion: {question}\nAnswer: "

        mt_inputs = self.mt_tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        with torch.no_grad():
            mt_inputs = {k: v.to(self.device) for k, v in mt_inputs.items()}
            out = self.mt_model(**mt_inputs)
            x_source = out.last_hidden_state


        context = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(answer, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if context["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            context["input_ids"] = context["input_ids"][:, :self.max_length - answer["input_ids"].shape[1]]
            context["attention_mask"] = context["attention_mask"][:, :self.max_length - answer["input_ids"].shape[1]]

        # train on context + answer
        labels = torch.cat([context["input_ids"], answer["input_ids"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.max_length - context["input_ids"].shape[1]

        if pad_length > 0:
            context["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), context["input_ids"]], dim=-1)
            context["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), context["attention_mask"]], dim=-1)
        
        pad_length = self.max_length - labels.shape[1]

        if pad_length > 0:
            labels = torch.cat([torch.full((1, pad_length), -100, dtype=torch.long), labels], dim=-1)
        
        return {
            "input_ids": context["input_ids"].squeeze(0),
            "attention_mask": context["attention_mask"].squeeze(0),
            "x_source": x_source.squeeze(0),
            "labels": labels.squeeze(0),
        }
    
    def __getitem_train__(self, idx):
        question = self.data["question"][idx]
        context = self.data["context"][idx]
        answer = self.data["answers"][idx]["text"][0] 
        
        text = f"Context: {context}\nQuestion: {question}\nAnswer: "
        
        mt_inputs = self.mt_tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        with torch.no_grad():
            mt_inputs = {k: v.to(self.device) for k, v in mt_inputs.items()}
            out = self.mt_model(**mt_inputs)
            x_source = out.last_hidden_state

        context = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        answer = self.tokenizer(answer, max_length=self.max_length-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        answer["input_ids"] = torch.cat([answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        answer["attention_mask"] = torch.cat([answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if context["input_ids"].shape[1] + answer["input_ids"].shape[1] > self.max_length:
            context["input_ids"] = context["input_ids"][:, :self.max_length - answer["input_ids"].shape[1]]
            context["attention_mask"] = context["attention_mask"][:, :self.max_length - answer["input_ids"].shape[1]]
        
        # concat

        # train on context + answer
        labels = torch.cat([torch.full_like(context["input_ids"], -100), answer["input_ids"]], dim=-1)
        context["input_ids"] = torch.cat([context["input_ids"], answer["input_ids"]], dim=-1)
        context["attention_mask"] = torch.cat([context["attention_mask"], answer["attention_mask"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.max_length - context["input_ids"].shape[1]

        if pad_length > 0:
            context["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), context["input_ids"]], dim=-1)
            context["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), context["attention_mask"]], dim=-1)
            labels = torch.cat([ torch.full((1, pad_length), -100, dtype=torch.long), labels], dim=-1)
        
        return {
            "input_ids": context["input_ids"].squeeze(0),
            "attention_mask": context["attention_mask"].squeeze(0),
            "x_source": x_source.squeeze(0),
            "labels": labels.squeeze(0),
        }

class CLM_QAFusionDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, data_target, tokenizer, args, is_testset=False):
        self.data_source = data_source
        self.data_target = data_target
        self.tokenizer = tokenizer
        self.args = args
        self.is_testset = is_testset
    
    def __len__(self):
        return len(self.data_target["question"])
    
    def __getitem__(self, idx):
        if self.is_testset:
            return self.__getitem_test__(idx)
        else:
            return self.__getitem_train__(idx)
    
    def __getitem_test__(self, idx):
        source_question = self.data_source["question"][idx]
        source_context = self.data_source["context"][idx]
        # source_answer = self.data_source["answers"][idx]["text"][0]

        target_question = self.data_target["question"][idx]
        target_context = self.data_target["context"][idx]
        target_answer = self.data_target["answers"][idx]["text"][0]

        source_text = f"Context: {source_context}\nQuestion: {source_question}\nAnswer: "
        target_text = f"Context: {target_context}\nQuestion: {target_question}\nAnswer: "

        source_context = self.tokenizer(source_text, max_length=self.args["max_length"], truncation=True, padding=False, return_tensors="pt")
        target_context = self.tokenizer(target_text, max_length=self.args["max_length"], truncation=True, padding=False, return_tensors="pt")
        target_answer = self.tokenizer(target_answer, max_length=self.args["max_length"]-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        target_answer["input_ids"] = torch.cat([target_answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        target_answer["attention_mask"] = torch.cat([target_answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if target_context["input_ids"].shape[1] + target_answer["input_ids"].shape[1] > self.args["max_length"]:
            target_context["input_ids"] = target_context["input_ids"][:, :self.args["max_length"] - target_answer["input_ids"].shape[1]]
            target_context["attention_mask"] = target_context["attention_mask"][:, :self.args["max_length"] - target_answer["input_ids"].shape[1]]
            
        if source_context["input_ids"].shape[1] + target_answer["input_ids"].shape[1] > self.args["max_length"]:
            source_context["input_ids"] = source_context["input_ids"][:, :self.args["max_length"] - target_answer["input_ids"].shape[1]]
            source_context["attention_mask"] = source_context["attention_mask"][:,:self.args["max_length"] - target_answer["input_ids"].shape[1]]

        # add target context to labels
        labels = torch.cat([target_context["input_ids"], target_answer["input_ids"]], dim=-1)

        # padding to max_length (left padding)
        pad_length_source = self.args["max_length"] - source_context["input_ids"].shape[1]
        pad_length_target = self.args["max_length"] - target_context["input_ids"].shape[1]
        pad_length_labels = self.args["max_length"] - labels.shape[1]

        if pad_length_source > 0:
            source_context["input_ids"] = torch.cat([torch.full((1, pad_length_source), self.tokenizer.pad_token_id, dtype=torch.long), source_context["input_ids"]], dim=-1)
            source_context["attention_mask"] = torch.cat([torch.full((1, pad_length_source), 0, dtype=torch.long), source_context["attention_mask"]], dim=-1)

        if pad_length_target > 0:
            target_context["input_ids"] = torch.cat([torch.full((1, pad_length_target), self.tokenizer.pad_token_id, dtype=torch.long), target_context["input_ids"]], dim=-1)
            target_context["attention_mask"] = torch.cat([torch.full((1, pad_length_target), 0, dtype=torch.long), target_context["attention_mask"]], dim=-1)
            
        if pad_length_labels > 0:
            labels = torch.cat([torch.full((1, pad_length_labels), -100, dtype=torch.long), labels], dim=-1)

        return {
            "input_ids_source": source_context["input_ids"].squeeze(0).long(),
            "attention_mask_source": source_context["attention_mask"].squeeze(0).long(),
            "input_ids_target": target_context["input_ids"].squeeze(0).long(),
            "attention_mask_target": target_context["attention_mask"].squeeze(0).long(),
            "labels": labels.squeeze(0).long(),
        }

    def __getitem_train__(self, idx):
        source_question = self.data_source["question"][idx]
        source_context = self.data_source["context"][idx]

        target_question = self.data_target["question"][idx]
        target_context = self.data_target["context"][idx]
        target_answer = self.data_target["answers"][idx]["text"][0]

        source_text = f"Context: {source_context}\nQuestion: {source_question}\nAnswer: " 
        target_text = f"Context: {target_context}\nQuestion: {target_question}\nAnswer: "

        source_context = self.tokenizer(source_text, max_length=self.args["max_length"], truncation=True, padding=False, return_tensors="pt")
        target_context = self.tokenizer(target_text, max_length=self.args["max_length"], truncation=True, padding=False, return_tensors="pt")
        target_answer = self.tokenizer(target_answer, max_length=self.args["max_length"]-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        target_answer["input_ids"] = torch.cat([target_answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        target_answer["attention_mask"] = torch.cat([target_answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if target_context["input_ids"].shape[1] + target_answer["input_ids"].shape[1] > self.args["max_length"]:
            target_context["input_ids"] = target_context["input_ids"][:, :self.args["max_length"] - target_answer["input_ids"].shape[1]]
            target_context["attention_mask"] = target_context["attention_mask"][:, :self.args["max_length"] - target_answer["input_ids"].shape[1]]
        
        if source_context["input_ids"].shape[1] + target_answer["input_ids"].shape[1] > self.args["max_length"]:
            source_context["input_ids"] = source_context["input_ids"][:, :self.args["max_length"] - target_answer["input_ids"].shape[1]]
            source_context["attention_mask"] = source_context["attention_mask"][:,:self.args["max_length"] - target_answer["input_ids"].shape[1]]

        # concat

        # train on context + answer
        labels = torch.cat([torch.full_like(target_context["input_ids"], -100), target_answer["input_ids"]], dim=-1)
        target_context["input_ids"] = torch.cat([target_context["input_ids"], target_answer["input_ids"]], dim=-1)
        target_context["attention_mask"] = torch.cat([target_context["attention_mask"], target_answer["attention_mask"]], dim=-1)
        # source_context["input_ids"] = torch.cat([source_context["input_ids"], source_answer["input_ids"]], dim=-1)
        # source_context["attention_mask"] = torch.cat([source_context["attention_mask"], target_answer["attention_mask"]], dim=-1)

        # padding to max_length (left padding)
        pad_length_source = self.args["max_length"] - source_context["input_ids"].shape[1]
        pad_length_target = self.args["max_length"] - target_context["input_ids"].shape[1]

        if pad_length_source > 0:
            source_context["input_ids"] = torch.cat([torch.full((1, pad_length_source), self.tokenizer.pad_token_id, dtype=torch.long), source_context["input_ids"]], dim=-1)
            source_context["attention_mask"] = torch.cat([torch.full((1, pad_length_source), 0, dtype=torch.long), source_context["attention_mask"]], dim=-1)
        
        if pad_length_target > 0:
            target_context["input_ids"] = torch.cat([torch.full((1, pad_length_target), self.tokenizer.pad_token_id, dtype=torch.long), target_context["input_ids"]], dim=-1)
            target_context["attention_mask"] = torch.cat([torch.full((1, pad_length_target), 0, dtype=torch.long), target_context["attention_mask"]], dim=-1)
            labels = torch.cat([ torch.full((1, pad_length_target), -100, dtype=torch.long), labels], dim=-1)
        
        return {
            "input_ids_source": source_context["input_ids"].squeeze(0).long(),
            "attention_mask_source": source_context["attention_mask"].squeeze(0).long(),
            "input_ids_target": target_context["input_ids"].squeeze(0).long(),
            "attention_mask_target": target_context["attention_mask"].squeeze(0).long(),
            "labels": labels.squeeze(0).long(),
        } 

class CLM_QAInputFusionDataset(torch.utils.data.Dataset):
    def __init__(self, data_source, data_target, tokenizer, args, is_testset=False):
        self.data_source = data_source
        self.data_target = data_target
        self.tokenizer = tokenizer
        self.args = args
        self.is_testset = is_testset
    
    def __len__(self):
        return len(self.data_target["question"])
    
    def __getitem__(self, idx):
        if self.is_testset:
            return self.__getitem_test__(idx)
        else:
            return self.__getitem_train__(idx)
    
    def __getitem_test__(self, idx):
        source_question = self.data_source["question"][idx]
        source_context = self.data_source["context"][idx]
        # source_answer = self.data_source["answers"][idx]["text"][0]

        target_question = self.data_target["question"][idx]
        target_context = self.data_target["context"][idx]
        target_answer = self.data_target["answers"][idx]["text"][0]

        text = f"Context: {source_context}\nQuestion: {source_question}\n\nContext: {target_context}\nQuestion: {target_question}Answer: "

        context = self.tokenizer(text, max_length=self.args["max_length"], truncation=True, padding=False, return_tensors="pt")
        target_answer = self.tokenizer(target_answer, max_length=self.args["max_length"]-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        target_answer["input_ids"] = torch.cat([target_answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        target_answer["attention_mask"] = torch.cat([target_answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if context["input_ids"].shape[1] + target_answer["input_ids"].shape[1] > self.args["max_length"]:
            context["input_ids"] = context["input_ids"][:, :self.args["max_length"] - target_answer["input_ids"].shape[1]]
            context["attention_mask"] = context["attention_mask"][:, :self.args["max_length"] - target_answer["input_ids"].shape[1]]

        # add target context to labels
        labels = torch.cat([context["input_ids"], target_answer["input_ids"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.args["max_length"] - context["input_ids"].shape[1]
        pad_length_labels = self.args["max_length"] - labels.shape[1]

        if pad_length > 0:
            context["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), context["input_ids"]], dim=-1)
            context["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), context["attention_mask"]], dim=-1)
            
        if pad_length_labels > 0:
            labels = torch.cat([torch.full((1, pad_length_labels), -100, dtype=torch.long), labels], dim=-1)

        return {
            "input_ids": context["input_ids"].squeeze(0),
            "attention_mask": context["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0).long(),
        }

    def __getitem_train__(self, idx):
        source_question = self.data_source["question"][idx]
        source_context = self.data_source["context"][idx]

        target_question = self.data_target["question"][idx]
        target_context = self.data_target["context"][idx]
        target_answer = self.data_target["answers"][idx]["text"][0]

        text = f"Context: {source_context}\nQuestion: {source_question}\n\nContext: {target_context}\nQuestion: {target_question}Answer: "

        context = self.tokenizer(text, max_length=self.args["max_length"], truncation=True, padding=False, return_tensors="pt")
        target_answer = self.tokenizer(target_answer, max_length=self.args["max_length"]-1, truncation=True, padding=False, return_tensors="pt", add_special_tokens=False)

        # add eos token
        target_answer["input_ids"] = torch.cat([target_answer["input_ids"], torch.tensor([self.tokenizer.eos_token_id]).unsqueeze(0)], dim=-1)
        target_answer["attention_mask"] = torch.cat([target_answer["attention_mask"], torch.tensor([1]).unsqueeze(0)], dim=-1)

        # truncate context to fit context + answer in max_length
        if context["input_ids"].shape[1] + target_answer["input_ids"].shape[1] > self.args["max_length"]:
            context["input_ids"] = context["input_ids"][:, :self.args["max_length"] - target_answer["input_ids"].shape[1]]
            context["attention_mask"] = context["attention_mask"][:, :self.args["max_length"] - target_answer["input_ids"].shape[1]]

        # concat

        # train on context + answer
        labels = torch.cat([torch.full_like(context["input_ids"], -100), target_answer["input_ids"]], dim=-1)
        context["input_ids"] = torch.cat([context["input_ids"], target_answer["input_ids"]], dim=-1)
        context["attention_mask"] = torch.cat([context["attention_mask"], target_answer["attention_mask"]], dim=-1)

        # padding to max_length (left padding)
        pad_length = self.args["max_length"] - context["input_ids"].shape[1]

        if pad_length > 0:
            context["input_ids"] = torch.cat([torch.full((1, pad_length), self.tokenizer.pad_token_id, dtype=torch.long), context["input_ids"]], dim=-1)
            context["attention_mask"] = torch.cat([torch.full((1, pad_length), 0, dtype=torch.long), context["attention_mask"]], dim=-1)
            labels = torch.cat([ torch.full((1, pad_length), -100, dtype=torch.long), labels], dim=-1)
        
        return {
            "input_ids": context["input_ids"].squeeze(0),
            "attention_mask": context["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0).long(),
        } 
        
# load dataset configuration
def load_data(args, lang):
    if args["translate-train"] and (lang == args["target_lang"]):

        print("Loading MT training data and HF test data:", args["path_mt"])

        train = pd.read_parquet(f"{args['path_mt']}/train_{args['source_lang']}-{args['target_lang']}.parquet")
        train = train[["context_translated", "question_translated", "answers_translated"]].rename(columns={"context_translated": "context", "question_translated": "question", "answers_translated": "answers"}).to_dict(orient="list")

        val = {k: v[:20] for k, v in train.items()} # not used for checkpoint selection (debugging)

        test = pd.read_parquet(os.path.join(args["path_data"], args["task"], "validation.parquet"))
        test = test[test["lang"] == lang].reset_index(drop=True).to_dict(orient="list")
    
    elif args["translate-train"] and (lang == args["source_lang"]):
        assert lang == "en", "Only support en as source language"

        # load original source language data
        print("Source: Loading HF train data and MT test data", args["path_mt_test"])

        train = pd.read_parquet(os.path.join(args["path_data"], args["task"], "train.parquet"))
        train = train[train["lang"] == lang].reset_index(drop=True).to_dict(orient="list")

        val = {k: v[:20] for k, v in train.items()} # not used for checkpoint selection (debugging)

        # load translated test data
        test = pd.read_parquet(f"{args['path_mt_test']}/test_{args['target_lang']}-{args['source_lang']}.parquet")
        test = test[["context_translated", "question_translated", "answers_translated"]].rename(columns={"context_translated": "context", "question_translated": "question", "answers_translated": "answers"}).to_dict(orient="list")
    
    elif args["translate-test"]:
        print("Loading MT data:", args["path_mt"])

        train = pd.read_parquet(f"{args['path_mt']}/train_{args['target_lang']}-{args['source_lang']}.parquet")
        test = pd.read_parquet(f"{args['path_mt']}/test_{args['target_lang']}-{args['source_lang']}.parquet")

        train = train[["context_translated", "question_translated", "answers_translated"]].rename(columns={"context_translated": "context", "question_translated": "question", "answers_translated": "answers"}).to_dict(orient="list")

        val = {k: v[:20] for k, v in train.items()} # not used for checkpoint selection (debugging)

        test = test[["context_translated", "question_translated", "answers_translated"]].rename(columns={"context_translated": "context", "question_translated": "question", "answers_translated": "answers"}).to_dict(orient="list")
    
    else:
        print("Loading original data:", os.path.join(args["path_data"], args["task"]))

        train = pd.read_parquet(os.path.join(args["path_data"], args["task"], "train.parquet"))
        train = train[train["lang"] == lang].reset_index(drop=True).to_dict(orient="list")
        val = {k: v[:20] for k, v in train.items()}
        test = pd.read_parquet(os.path.join(args["path_data"], args["task"], "validation.parquet"))
        test = test[test["lang"] == lang].reset_index(drop=True).to_dict(orient="list")

    # remove all '"' which are used for label projection
    for k in ["context", "question"]:
        train[k] = [re.sub(r'"', '', i) for i in train[k]]
        val[k] = [re.sub(r'"', '', i) for i in val[k]]
        test[k] = [re.sub(r'"', '', i) for i in test[k]]

    return train, val, test

# get regular / fusion dataloader
def get_dataloader(train, val, test, tokenizer, args, **kwargs):

    if (args["fusion_fn"] is not None):
        # Fusion Datasets
        if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]:
            if args["fuse_mt"]:
                # MT Fusion Datasets (CLM)
                ds_train = MTFusionCLM_QADataset(data=train, tokenizer=tokenizer, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"], args=args)
                ds_val = MTFusionCLM_QADataset(data=val, tokenizer=tokenizer, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"], args=args, is_testset=True)
                ds_test = MTFusionCLM_QADataset(data=test, tokenizer=tokenizer, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"], args=args, is_testset=True)
            else:
                ds_train = CLM_QAFusionDataset(data_source=train["source"], data_target=train["target"], tokenizer=tokenizer, args=args)
                ds_val = CLM_QAFusionDataset(data_source=val["source"], data_target=val["target"], tokenizer=tokenizer, args=args, is_testset=True)
                ds_test = CLM_QAFusionDataset(data_source=test["source"], data_target=test["target"], tokenizer=tokenizer, args=args, is_testset=True)
        else:
            if args["fuse_mt"]:
                # MT Fusion Datasets
                ds_train = MTFusionQADataset(data=train, tokenizer=tokenizer, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"], args=args)
                ds_val = MTFusionQADataset(data=val, tokenizer=tokenizer, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"], args=args)
                ds_test = MTFusionQADataset(data=test, tokenizer=tokenizer, mt_model=kwargs["mt_model"], mt_tokenizer=kwargs["mt_tokenizer"], args=args)
            else:
                ds_train = QAFusionDataset(data_source=train["source"], data_target=train["target"], tokenizer=tokenizer, args=args)
                ds_val = QAFusionDataset(data_source=val["source"], data_target=val["target"], tokenizer=tokenizer, args=args)
                ds_test = QAFusionDataset(data_source=test["source"], data_target=test["target"], tokenizer=tokenizer, args=args)
    elif args["input_fusion"]:
        if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]: 
            ds_train = CLM_QAInputFusionDataset(data_source=train["source"], data_target=train["target"], tokenizer=tokenizer, args=args)
            ds_val = CLM_QAInputFusionDataset(data_source=val["source"], data_target=val["target"], tokenizer=tokenizer, args=args, is_testset=True)
            ds_test = CLM_QAInputFusionDataset(data_source=test["source"], data_target=test["target"], tokenizer=tokenizer, args=args, is_testset=True)
        else:
            ds_train = QAInputFusionDataset(data_source=train["source"], data_target=train["target"], tokenizer=tokenizer, args=args)
            ds_val = QAInputFusionDataset(data_source=val["source"], data_target=val["target"], tokenizer=tokenizer, args=args)
            ds_test = QAInputFusionDataset(data_source=test["source"], data_target=test["target"], tokenizer=tokenizer, args=args)
    else:
        # Regular Datasets
        if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]: 
            ds_train = CLM_QADataset(data=train, tokenizer=tokenizer, args=args)
            ds_val = CLM_QADataset(data=val, tokenizer=tokenizer, args=args, is_testset=True)
            ds_test = CLM_QADataset(data=test, tokenizer=tokenizer, args=args, is_testset=True)
        else:
            ds_train = QADataset(data=train, tokenizer=tokenizer, args=args)
            ds_val = QADataset(data=val, tokenizer=tokenizer, args=args)
            ds_test = QADataset(data=test, tokenizer=tokenizer, args=args)

    return ds_train, ds_val, ds_test

def filter_qa_translations(*args):
    if len(args) == 1:
        return filter_qa_translations_mono(*args)
    elif len(args) == 2:
        return filter_qa_translations_multi(*args)
    else:
        raise NotImplementedError

def filter_qa_translations_mono(data):
    
        rm_idx = []
    
        # filter answer_start -1
        for i in range(len(data["answers"])):
            if data["answers"][i]["answer_start"][0] == -1:
                rm_idx.append(i)
                    
        # apply filter
        for k,v in data.items():
            data[k] = [v[i] for i in range(len(data["answers"])) if i not in rm_idx]
        
        print(f"Removed {len(rm_idx)} examples")
    
        return data

def filter_qa_translations_multi(source, target):

    rm_idx = []

    assert len(source["answers"]) == len(target["answers"]), print({"source": len(source["answers"]), "target": len(target["answers"])})

    # remove keys that are not in both source and target
    common = set(source.keys()).intersection(set(target.keys()))
    
    to_remove = [k for k in source.keys() if k not in common]
    for k in to_remove:
        source.pop(k)
    
    to_remove = [k for k in target.keys() if k not in common]
    for k in to_remove:
        target.pop(k)

    # filter answer_start -1
    for i in range(len(source["answers"])):
        if source["answers"][i]["answer_start"][0] == -1:
            rm_idx.append(i)
            continue
        if target["answers"][i]["answer_start"][0] == -1:
            rm_idx.append(i)
            continue
        # filter empty answers
        if len(target["answers"][i]["text"][0]) < 1:
            rm_idx.append(i)
            continue
    
    # apply filter
    for k,v in source.items():
        source[k] = [v[i] for i in range(len(source["answers"])) if i not in rm_idx]
        target[k] = [target[k][i] for i in range(len(target["answers"])) if i not in rm_idx]
    
    print(f"Removed {len(rm_idx)} examples")

    return source, target

def get_compute_metrics_fn(args):
    if args["plm"] in ["llama3-8b", "llama3.1-8b", "gemma2-9b"]: # "llama3-8b", "llama3.1-8b", "mt5-xl"
        return compute_metrics_generate
    else:
        return compute_metrics_cls
    
def compute_metrics_cls(eval_preds):
    preds, (start_positions, end_positions) = eval_preds

    pred_start = preds[0].argmax(-1)
    pred_end = preds[1].argmax(-1)

    em = [1 if ((pred_start[i] == start_positions[i]) and (pred_end[i] == end_positions[i])) else 0 for i in range(len(start_positions))]

    return {"exact_match": sum(em) / len(em)}

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

    res = metric.compute(predictions=decoded_preds, references=decoded_labels, ignore_case=True, ignore_punctuation=True)

    return res

def evaluate(trainer, ds, tokenizer, args, prefix=None, **kwargs):
    trainer.model.eval()

    res = trainer.predict(ds).metrics

    # compute metrics        
    metrics = dict(res)

    if prefix is not None:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

    wandb.config.update(metrics)
    wandb.run.summary.update(metrics) 

    # save as json
    metrics.update(args)
    with open(os.path.join(args["output_dir"], args["run_name"], f"results_{prefix}.json"), "w") as f:
        json.dump(metrics, f)
    
    return metrics