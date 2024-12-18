from typing import Callable, List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FactScoreOutput:
    atomic_facts: List[str]
    scores: List[Any]
    aggregated_score: Any


class FactScore:
    def __init__(
        self,
        atomic_fact_generator: Callable[[str], List[str]],
        knowledge_source: Callable[[str, str], List[Dict[str, str]]],
        scorer: Callable[[str, str, List[Dict[str, str]]], Any],
        aggregate_fn: Optional[Callable[[List[Any]], Any]]=None,
    ) -> None:
        self.atomic_fact_generator = atomic_fact_generator
        self.knowledge_source = knowledge_source
        self.scorer = scorer
        self.aggregate_fn = aggregate_fn

    def evaluate(self, topic: str, generation: str) -> FactScoreOutput:
        scores =[]
        atomic_facts = self.atomic_fact_generator(generation)
        for atomic_fact in atomic_facts:
            knowledges = self.knowledge_source(topic, atomic_fact)
            score = self.scorer(topic, atomic_fact, knowledges)
            scores.append(score)
        aggregated = self.aggregate_fn(scores) if self.aggregate_fn is not None else None
        return FactScoreOutput(
            atomic_facts=atomic_facts,
            scores=scores,
            aggregated_score=aggregated,
        )
