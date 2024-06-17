from typing import Set
from tabulate import tabulate
from dataclasses import dataclass, asdict
from pnpxai.detector.types import ModuleType


@dataclass
class RecommenderOutput:
    architecture: Set[ModuleType]
    explainers: list
    metrics: list

    def print_tabular(self):
        print(tabulate([
            [k, [v.__name__ for v in vs]]
            for k, vs in asdict(self).items()
        ]))