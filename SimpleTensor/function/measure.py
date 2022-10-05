from SimpleTensor.core import Node
from SimpleTensor.core import reduction_mean, reduction_sum


def mean_square_error(predict: Node, label: Node, reduction: str="mean"):
    __reductions__ = ["mean", "sum"]
    if reduction == "mean":
        return reduction_mean((predict - label) ** 2)

    elif reduction == "sum":
        return reduction_sum((predict - label) ** 2)

    else:
        raise Exception(f"reduction only receive {__reductions__}")
