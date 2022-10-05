import numpy as np

from SimpleTensor.runtime import activate_func
from SimpleTensor.core import Operation, Node


@activate_func('add')
def add(a: int, b: int):
    return a + b


@activate_func('sigmod')
class sigmod(Operation):
    def __init__(self, x: Node, node_name: str = ''):
        super(sigmod, self).__init__(input_nodes=[x], node_name=node_name)

    def compute(self, x_v: np.ndarray):
        return 1 / (1 + np.exp(-1 * x_v))

