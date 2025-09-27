import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from inspect_ai.dataset import MemoryDataset, Sample

SPLIT_TYPE = Literal["train", "dev", "test"]


@dataclass
class QuALITYExample:
    metadata: dict[str, int | str]
    article: str
    question: str
    answer: dict[str, str]

    def to_sample(self) -> Sample:
        choices_letters: list[str] = ["A", "B"]
        gold_idx = random.randint(0, 1)
        if gold_idx == 0:
            choices = [self.answer["gold"], self.answer["best_distractor"]]
        else:
            choices = [self.answer["best_distractor"], self.answer["gold"]]

        return Sample(
            metadata={
                "answers": self.answer,
                "letters": choices_letters,
                "article": self.article,
                "question": self.question,
                "choices": choices,
                "target": choices_letters[gold_idx],
                **self.metadata,
            },
            input=self.question,
            target=choices_letters[gold_idx],
            choices=choices,
        )


class QuALITY:
    """QuALITY dataset with pre-processing described in the Debating with More
    Persuasive LLMs Leads to More Truthful Answers paper."""

    def __init__(self, data_dir: Path, random_seed: int = 25) -> None:
        if data_dir.is_dir() and not any(data_dir.iterdir()):
            raise FileNotFoundError(
                f"{data_dir=} is empty. Download data before start!"
            )
        self.random_seed = random_seed
        self.data_dir = data_dir
        self.split: dict[str, list[QuALITYExample]] = {}

    def prepare_datasets(self) -> None:
        for split in ["train"]:
            self._prepare_dataset_per_split(split)

    def _prepare_dataset_per_split(self, split: str) -> None:
        with open(self.data_dir / f"QuALITY.v1.0.1.htmlstripped.{split}") as f:
            data = [json.loads(line) for line in f]
        filtered_dataset = []
        for example in data:
            if not self._filter_gutenberg_short_stories(example):
                continue
            for question in example["questions"]:
                if not self._filter_question_based_on_annotation(question):
                    continue
                # for some weird reason, they are indexed from 1
                gold_answer = question["options"][question["gold_label"] - 1]
                best_distractor = self._get_best_distractor(question)
                prep_example = QuALITYExample(
                    metadata={
                        "article_id": example["article_id"],
                        "set_unique_id": example["set_unique_id"],
                        "title": example["title"],
                        "year": example["year"],
                        "author": example["author"],
                        "topic": example["topic"],
                    },
                    article=example["article"],
                    question=question["question"],
                    answer={
                        "gold": gold_answer,
                        "best_distractor": best_distractor,
                    },
                )
                filtered_dataset.append(prep_example)

        self.split[split] = filtered_dataset

    def _filter_gutenberg_short_stories(self, example):
        """[...]We use only questions from the project Gutenberg short science[...]"""
        return "short stories" in example["topic"].lower()

    def _filter_question_based_on_annotation(self, question: dict) -> bool:
        "1. 100% of untimed annotators chose the correct answer"
        true_label = question["gold_label"]
        is_annot_answers_true = [
            annot["untimed_answer"] == true_label for annot in question["validation"]
        ]
        coherent_answers: bool = all(is_annot_answers_true)
        "2. Less than 50% of timed annotators chose the correct answer"
        speed_annot_answers = [
            annot["speed_answer"] == true_label
            for annot in question["speed_validation"]
        ]
        timed_annot_wrong_half_time: bool = (
            sum(speed_annot_answers) / len(speed_annot_answers)
        ) < 0.5
        "3.All untimed annotators agree that the question is answerable and unambiguous"
        untimed_answerable_unambig: bool = all(
            [
                annot["untimed_eval1_answerability"] == 1
                for annot in question["validation"]
            ]
        )
        "4. Average ”context required” rating from untimed annotators is at least 1.5"
        avg_context_more_eq_1_5: bool = bool(
            np.mean(
                [annot["untimed_eval2_context"] for annot in question["validation"]]
            )
            >= 1.5
        )
        "5. Writer label matches the gold label"
        writer_label_coherent: bool = question["writer_label"] == true_label
        return (
            coherent_answers
            and timed_annot_wrong_half_time
            and untimed_answerable_unambig
            and avg_context_more_eq_1_5
            and writer_label_coherent
        )

    def _get_best_distractor(self, question: dict) -> str:
        best_distractor_candidates = [
            annot["untimed_best_distractor"] for annot in question["validation"]
        ]
        best_distractor_idx = max(
            best_distractor_candidates,
            key=lambda x: best_distractor_candidates.count(x),
        )
        best_distractor_idx -= 1  # for some weird reason, they are indexed from 1
        return question["options"][best_distractor_idx]

    def get_memory_dataset(self, split: SPLIT_TYPE) -> MemoryDataset:
        if self.split.get(split) is None:
            self._prepare_dataset_per_split(split)
        converted_dataset = [example.to_sample() for example in self.split[split]]
        return MemoryDataset(converted_dataset)
