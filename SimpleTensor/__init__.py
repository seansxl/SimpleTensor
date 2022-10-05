from SimpleTensor.core import Node, Variable, Placeholder, Operation, _default_graph, Session
from SimpleTensor.function import dnn
from SimpleTensor.function.measure import mean_square_error
from SimpleTensor.optimizer import SGD
# register
import SimpleTensor.function.activate
import SimpleTensor.function.gradient

from SimpleTensor.runtime import gradient_func
