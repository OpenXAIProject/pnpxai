from enum import Enum


class APIItems(Enum):
    MESSAGE = 'message'
    DATA = 'data'
    ERRORS = 'errors'

    ID = 'id'
    NAME = 'name'
    SOURCE = 'source'
    TARGET = 'target'

    EXPLANATIONS = 'explanations'
    EVALUATION = 'evaluation'
    INPUT = 'input'
    EXPLAINER = 'explainer'

    NODES = 'nodes'
    EDGES = 'edges'

    EXPERIMENTS = 'experiments'
    EXPLAINERS = 'explainers'
