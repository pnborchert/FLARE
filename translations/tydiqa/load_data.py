from datasets import load_dataset
import pandas as pd

LANG_MAPPING = {
    "arabic":"ar",
    "finnish": "fi",
    "russian":"ru",
    "indonesian":"ind",
    "telugu": "tel",
    "english":"en",
    "swahili":"sw",
    "bengali":"ben",
    "korean":"ko",
}

dataset = load_dataset("tydiqa", "secondary_task")

for split in ["train", "validation"]:
    data = dataset[split].to_pandas()
    data["lang"] = data["id"].str.split("-").str[0]
    data["lang"] = data["lang"].map(LANG_MAPPING)

    # save as parquet
    data.to_parquet(f"{split}.parquet", index=False)
