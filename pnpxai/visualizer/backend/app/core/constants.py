from enum import Enum


class APIItems(Enum):
    MESSAGE = 'message'
    DATA = 'data'
    ERRORS = 'errors'
    
    TRACE = 'trace'

    ID = 'id'
    KEY = 'key'
    OPCODE = 'opcode'
    OPERATOR = 'operator'
    NAME = 'name'
    VALUE = 'value'
    SOURCE = 'source'
    TARGET = 'target'
    OUTPUTS = 'outputs'

    EXPLANATIONS = 'explanations'
    RANK = 'rank'
    EVALUATION = 'evaluation'
    INPUT = 'input'
    EXPLAINER = 'explainer'

    NODES = 'nodes'
    EDGES = 'edges'

    EXPERIMENTS = 'experiments'
    EXPLAINERS = 'explainers'
    METRICS = 'metrics'
