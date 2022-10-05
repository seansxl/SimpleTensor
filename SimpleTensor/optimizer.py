import abc

from SimpleTensor import Operation, Placeholder, Variable
from SimpleTensor import runtime
from collections import deque


def backwards(op_node: Operation):
    grad_table = {}
    grad_table[op_node] = 1.
    visit_nodes = set()
    queue = deque()
    visit_nodes.add(op_node)
    queue.append(op_node)

    while len(queue) > 0:
        cur_node = queue.popleft()
        if cur_node != op_node and not (isinstance(cur_node, Placeholder)):
            grad_table[cur_node] = 0.
            for next_node in cur_node.next_nodes:
                grad_loss_wrt_next_node = grad_table[next_node]
                next_node_op_name = next_node.__class__.__name__
                # print(next_node_op_name, runtime.gradient_func.items())
                gradient_func = runtime.gradient_func[next_node_op_name]
                grad_loss_wrt_cur_node = gradient_func(next_node, grad_loss_wrt_next_node)

                if len(next_node.input_nodes) == 1:
                    grad_table[cur_node] += grad_loss_wrt_cur_node[0]
                else:
                    cur_node_in_next_node_index = next_node.input_nodes.index(cur_node)
                    grad_table[cur_node] += grad_loss_wrt_cur_node[cur_node_in_next_node_index]

        if isinstance(cur_node, Operation):
            for input_node in cur_node.input_nodes:
                if input_node not in visit_nodes:
                    visit_nodes.add(input_node)
                    queue.append(input_node)

    return grad_table


class Optimizer(abc.ABC):
    def __init__(self, learning_rate: float = 1e-3):
        self.learning_rate = learning_rate

    @abc.abstractmethod
    def minimize(self, loss_node: Operation):
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-5):
        super(SGD, self).__init__(learning_rate=learning_rate)
        print(learning_rate)

    def minimize(self, loss_node: Operation):
        lr = self.learning_rate
        grad_table = backwards(op_node=loss_node)
        for node in grad_table:
            if isinstance(node, Variable):
                grad = grad_table[node]
                node.data -= lr * grad

        runtime.grad_table = grad_table
        return grad_table