from datasets import load_dataset

from nemo_reinforcer.data.hf_datasets.interfaces import HfDataset


## assumptions about DPO dataset:
## json files should have the following keys:
## "prompt"
## "chosen_response"
## "rejected_response"
class DPODataset(HfDataset):
    def __init__(self, train_data_path: str, val_data_path: str):
        ## TODO: assuming for now that data has been split into train and val
        ## as an offline preprocessing step

        ## TODO: update the keys to match with what's expected from apply_chat_template
        ## we need to do this outisde of the data class because we want to keep
        ## chosen and rejected responses for a given prompt together when shuffling
        self.formatted_ds = {
            "train": load_dataset("json", data_files=train_data_path),
            "validation": load_dataset("json", data_files=val_data_path),
        }
        super().__init__(
            dataset_name="dpo",
            ## no custom template. Assume we use tokenizer's template
            # custom_template=COMMON_CHAT_TEMPLATES.simple_role_header,
        )
