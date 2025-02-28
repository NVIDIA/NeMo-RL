import pytest

from transformers import AutoTokenizer
from nemo_reinforcer.data.hf_datasets.squad import SquadDataset


@pytest.mark.skip(reason="dataset download is flaky")
def test_squad_dataset():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    squad_dataset = SquadDataset()

    # check that the dataset is formatted correctly
    for example in squad_dataset.formatted_ds["train"].take(5):
        assert "messages" in example
        assert len(example["messages"]) == 3

        assert example["messages"][0]["role"] == "system"
        assert example["messages"][1]["role"] == "user"
        assert example["messages"][2]["role"] == "assistant"

        ## check that applying chat template works as expected
        default_templated = tokenizer.apply_chat_template(
            example["messages"],
            chat_template=squad_dataset.task_spec.custom_template,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )

        assert default_templated == (
            "Context: "
            + example["messages"][0]["content"]
            + " Question: "
            + example["messages"][1]["content"]
            + " Answer: "
            + example["messages"][2]["content"]
        )
