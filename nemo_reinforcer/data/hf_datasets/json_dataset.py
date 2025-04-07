from datasets import load_dataset
from nemo_reinforcer.data.hf_datasets.interfaces import HfDataset


class JsonDataset(HfDataset):
    def __init__(
        self,
        train_ds_path: str,
        val_ds_path: str,
        input_key: str = "input",
        output_key: str = "output",
    ):
        train_original_dataset = load_dataset("json", data_files=train_ds_path)["train"]
        val_original_dataset = load_dataset("json", data_files=val_ds_path)["train"]
        formatted_train_dataset = train_original_dataset.map(self.add_messages_key)
        formatted_val_dataset = val_original_dataset.map(self.add_messages_key)

        ## just duplicating the dataset for train and validation for simplicity
        self.formatted_ds = {
            "train": formatted_train_dataset,
            "validation": formatted_val_dataset,
        }

        self.input_key = input_key
        self.output_key = output_key

        super().__init__(
            "json_dataset",
        )

    def add_messages_key(self, example):
        return {
            "messages": [
                {"role": "user", "content": example[self.input_key]},
                {"role": "assistant", "content": example[self.output_key]},
            ]
        }
