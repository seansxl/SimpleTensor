import numpy as np

from SimpleTensor import runtime
from SimpleTensor.core import Variable, Node

# np.random.seed(1)


class Linear:
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 bias: bool = True,
                 act: str = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act

        self.w = Variable(np.random.randn(input_dim, output_dim), node_name='w')
        if bias:
            self.b = Variable(np.random.randn(1, output_dim), node_name='b')

    def __call__(self, x: Node):
        out = x @ self.w + self.b
        if self.act:
            act_func = runtime.activate_func[self.act]
            return act_func(out)
        else:
            return out
