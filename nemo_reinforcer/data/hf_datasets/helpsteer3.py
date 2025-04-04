from datasets import load_dataset
from nemo_reinforcer.data.hf_datasets.interfaces import HfDataset


def format_helpsteer3(data):
    context = data["context"]
    response_1 = data["response1"]
    response_2 = data["response2"]
    overall_preference = data["overall_preference"]

    return {
        "prompt": data["context"],
        "chosen_response": response_1 if overall_preference < 0 else response_2,
        "rejected_response": response_2 if overall_preference < 0 else response_1,
    }


class HelpSteer3Dataset(HfDataset):
    def __init__(self):
        ds = load_dataset("nvidia/HelpSteer3", "preference")
        self.formatted_ds = ds.map(format_helpsteer3)

        super().__init__(
            dataset_name="HelpSteer3",
            custom_template=None,  ## use tokenizer's template
        )
