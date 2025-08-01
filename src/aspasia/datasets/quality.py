from typing import Any
import json
from inspect_ai.dataset import hf_dataset
from pathlib import Path

class QuALITY:
    """
    QuALITY dataset with pre-processing described in the 
    Debating with More Persuasive LLMs Leads to More Truthful Answers paper.
    """

    def __init__(self, data_dir:Path) -> None:
        if data_dir.is_dir() and not any(data_dir.iterdir()):
            raise FileNotFoundError(f"{data_dir=} is empty. Download data before start!")
        self.data_dir = data_dir

    def prepare_dataset(self) -> None:
        for split in ["train", "val", "test"]:
            self.prepare_dataset_per_split(split)
    
    def prepare_dataset_per_split(self, split: str) -> None:
        with open(self.data_dir/ f"QuALITY.v1.0.1.htmlstripped.{split}") as f:
            data = [json.loads(line) for line in f]
        print(data[0])


    @property
    def dataset(self) -> Any:
        return 1



if __name__ == "__main__":
    data_dir = Path("/Users/tsimur.hadeliya/code/aspasia/data")
    dataset =QuALITY(data_dir=data_dir)
    dataset.prepare_dataset()