import tempfile
import json


def make_dpo_dataset():
    train_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    val_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)

    # Write train data
    train_data = [
        {"context": "What is 2+2?", "chosen": "4", "rejected": "5"},
        {"context": "What is 3*3?", "chosen": "9", "rejected": "6"},
    ]
    for item in train_data:
        lines = train_file.write(json.dumps(item) + "\n")
    train_file.flush()

    # Write validation data
    val_data = [
        {"context": "What is 4+4?", "chosen": "8", "rejected": "7"},
        {"context": "What is 5*5?", "chosen": "25", "rejected": "20"},
    ]
    for item in val_data:
        lines = val_file.write(json.dumps(item) + "\n")
    val_file.flush()

    return train_file, val_file
