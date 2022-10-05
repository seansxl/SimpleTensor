import abc
import numpy as np
from typing import Union, List
from pprint import pprint

_default_graph = []


class Node: ...


class Node:
    def __init__(self, node_name: str = ''):
        self.next_nodes = []
        self.data = None
        self.node_name = node_name
        _default_graph.append(self)

    def __add__(self, node: Node):
        return add(self, node)

    def __sub__(self, node: Node):
        return sub(self, node)

    def __mul__(self, node: Node):
        return multiply(self, node)

    def __matmul__(self, node: Node):
        return matmul(self, node)

    def __pow__(self, node: Node, modulo=None):
        return pow(self, node, modulo)

    @property
    def numpy(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the session first!")
        return np.array(self.data)

    @property
    def shape(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the session first!")
        return np.array(self.data).shape

    def __str__(self):
        return f"{self.__class__.__name__}({self.data})"


class Placeholder(Node):
    def __init__(self):
        super(Placeholder, self).__init__()


class Variable(Node):
    def __init__(self, init_value: Union[np.ndarray, list] = None, node_name: str = ''):
        super(Variable, self).__init__(node_name=node_name)
        self.data = init_value


class Operation(Node):
    def __init__(self, input_nodes: List[Node] = [], node_name: str = ''):
        super(Operation, self).__init__(node_name=node_name)
        self.input_nodes = input_nodes
        for node in input_nodes:
            node.next_nodes.append(self)

    @abc.abstractmethod
    def compute(self, *args): ...


class add(Operation):
    def __init__(self, x: Node, y: Node):
        super(add, self).__init__(input_nodes=[x, y])

    def compute(self, x_v: np.ndarray, y_v: np.ndarray):
        return x_v + y_v


class sub(Operation):
    def __init__(self, x: Node, y: Node):
        super(sub, self).__init__(input_nodes=[x, y])

    def compute(self, x_v: np.ndarray, y_v: np.ndarray):
        return x_v - y_v


class multiply(Operation):
    def __init__(self, x: Node, y: Node):
        super(multiply, self).__init__(input_nodes=[x, y])

    def compute(self, x_v: np.ndarray, y_v: np.ndarray):
        return x_v * y_v


class negative(Operation):
    def __init__(self, x: Node):
        super(negative, self).__init__(input_nodes=[x])

    def compute(self, x_v: np.ndarray):
        return -1. * x_v


class matmul(Operation):
    def __init__(self, x: Node, y: Node):
        super(matmul, self).__init__(input_nodes=[x, y])

    def compute(self, x_v: np.ndarray, y_v: np.ndarray):
        return x_v @ y_v


class pow(Operation):
    def __init__(self, x: Node, modulo: Union[int, float], node_name=""):
        super(pow, self).__init__(input_nodes=[x], node_name=node_name)
        self.modulo = modulo

    def compute(self, x_v: np.ndarray):
        return x_v ** self.modulo


class reduction_mean(Operation):
    def __init__(self, x: Node, axis: int = None, node_name: str = ""):
        super(reduction_mean, self).__init__(input_nodes=[x], node_name=node_name)
        self.axis = axis

    def compute(self, x_v : np.ndarray):
        return np.mean(x_v, axis=self.axis)


class reduction_sum(Operation):
    def __init__(self, x: Node, axis: int = None, node_name: str = ""):
        super(reduction_sum, self).__init__(input_nodes=[x], node_name=node_name)
        self.axis = axis

    def compute(self, x_v : np.ndarray):
        return np.sum(x_v, axis=self.axis)


# def Linear(input_dim : int, output_dim : int, bias : bool = True):
#     w = Variable(np.random.randn(input_dim, output_dim))
#     if bias:
#         b = Variable(np.random.randn(1, output_dim))
#         return lambda x: x @ w + b
#     else:
#         return lambda x: x @ w


class Session():

    def run(self, root_op: Operation, feed_dict: dict = {}):
        for node in _default_graph:
            if isinstance(node, Variable):
                node.data = np.array(node.data)
            elif isinstance(node, Placeholder):
                node.data = np.array(feed_dict[node])
            else:
                input_datas = [n.data for n in node.input_nodes]
                node.data = node.compute(*input_datas)
        return root_op


if __name__ == '__main__':
    # x = Placeholder()
    # # w = Variable(np.random.randn(13, 1))
    # # b = Variable(np.random.randn(1, 1))
    #
    # # out = x @ w + b
    # out = Linear(13, 1)(x)
    # print(out)
    # pprint(_default_graph)

    from runtime import activate_func

    print(activate_func.keys())
