from datasets import load_dataset
from nemo_reinforcer.data.hf_datasets.interfaces import HfDataset


class JsonDataset(HfDataset):
    def __init__(self, path: str):
        original_dataset = load_dataset("json", data_files=path)["train"]
        formatted_dataset = original_dataset.map(self.add_messages_key)

        ## just duplicating the dataset for train and validation for simplicity
        self.formatted_ds = {
            "train": formatted_dataset,
            "validation": formatted_dataset,
        }

        super().__init__(
            "json_dataset",
        )

    def add_messages_key(self, example):
        return {
            "messages": [
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": example["output"]},
            ]
        }
