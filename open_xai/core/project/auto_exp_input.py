from dataclasses import dataclass
from typing import Callable, Any
from open_xai.core._types import DataSource, Model


@dataclass
class AutoExplanationInput:
    model: Model
    data: DataSource
    question: str
    task: str
    input_extractor: Callable[[DataSource], Any] = lambda x: x[0]
    label_extractor: Callable[[DataSource], Any] = lambda x: x[1]

    def __post_init__(self):
        valid_questions = {'why', 'how'}
        valid_tasks = {'image', 'tabular'}

        assert self.question in valid_questions, ValueError(
            f"Invalid question: {self.question}. Choose from {', '.join(valid_questions)}."
        )
        assert self.task in valid_tasks, ValueError(
            f"Invalid task: {self.task}. Choose from {', '.join(valid_tasks)}."
        )
