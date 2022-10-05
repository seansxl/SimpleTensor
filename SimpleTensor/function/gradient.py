import numpy as np
from SimpleTensor.runtime import gradient_func
from SimpleTensor.core import Operation, Node


class Clip:
    PRECISE_LOW = 1e-127
    PRECISE_HIGH = 1e128
    EXP_PRECISE_LOW = -292.42
    EXP_RPECISE_HIGH = 294.73
    EXP = lambda x: np.exp(np.clip(x, Clip.EXP_PRECISE_LOW, Clip.EXP_RPECISE_HIGH))


def __get_grad_by_shape(node: Node, grad: np.ndarray):
    node_shape, grad_shape = node.shape, grad.shape
    if node_shape == grad_shape:
        return grad
    else:
        for axis, _ in enumerate(grad_shape):
            if grad_shape[axis] != node_shape[axis]:
                break
        return grad.mean(axis=axis).reshape(node_shape)


@gradient_func('negative')
def negative_backward(op_node: Operation, grad: np.ndarray):
    return np.array(-1 * grad)


def multiply_backward(op_node: Operation, grad: np.ndarray):
    x = op_node.input_nodes[0]


@gradient_func('sigmod')
def sigmod_backward(op_node: Operation, grad: np.ndarray):
    x = op_node.input_nodes[0].data
    ex = Clip.EXP(x)
    return [ex / ((1 + ex) ** 2) * grad]


@gradient_func('add')
def add_backward(op_node: Operation, grad: np.ndarray):
    # return [1. * grad, 1. * grad]
    return [
        1. * __get_grad_by_shape(op_node.input_nodes[0].data, grad),
        1. * __get_grad_by_shape(op_node.input_nodes[1].data, grad)
    ]

@gradient_func('sub')
def sub_backward(op_node: Operation, grad: np.ndarray):
    # return [1. * grad, -1. * grad]
    return [
        1. * __get_grad_by_shape(op_node.input_nodes[0].data, grad),
        -1. * __get_grad_by_shape(op_node.input_nodes[1].data, grad)
    ]

@gradient_func('matmul')
def matmul_backward(op_node: Operation, grad: np.ndarray):
    x = op_node.input_nodes[0].data
    y = op_node.input_nodes[1].data
    return [grad @ y.T, x.T @ grad]


@gradient_func('reduction_mean')
def reduce_mean_backward(op_node: Operation, grad: np.ndarray):
    grad_shape = op_node.input_nodes[0].shape
    multiplier = 1
    return [1. * np.ones(grad_shape) / multiplier * grad]

@gradient_func('pow')
def pow_backward(op_node: Operation, grad: np.ndarray):
    x = op_node.input_nodes[0].data
    modulo = op_node.modulo
    return [modulo * (x ** (modulo - 1)) * grad]

