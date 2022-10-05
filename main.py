import SimpleTensor as st
from pprint import pprint
from SimpleTensor.optimizer import backwards
from sklearn.datasets import load_boston
import numpy as np
import warnings
warnings.filterwarnings('ignore')

x, y = load_boston(return_X_y=True)
np.random.seed(123)
ratio = 0.8
sample_num = x.shape[0]
offline = int(sample_num * ratio)
indexes = np.arange(sample_num)
np.random.shuffle(indexes)

train_x, train_y = x[indexes[:offline]], y[indexes[:offline]].reshape([-1, 1])
test_x, test_y = x[indexes[offline:]], y[indexes[offline:]].reshape([-1, 1])
# print(train_x.shape, train_y.shape)
# print(test_x.shape, test_y.shape)

x = st.Placeholder()
y = st.Placeholder()

out1 = st.dnn.Linear(13, 8, act='sigmod')(x)
out2 = st.dnn.Linear(8, 4, act='sigmod')(out1)
out3 = st.dnn.Linear(4, 1)(out2)
loss = st.mean_square_error(predict=out3, label=y)
print(loss)
# pprint(st._default_graph)

session = st.Session()
optimizer = st.SGD(learning_rate=1e-3)
losses = []
for epoch in range(100):
    session.run(root_op=loss, feed_dict={x: train_x, y: train_y})
    losses.append(loss.numpy)
    optimizer.minimize(loss)
    print(f"\033[32m[Epoch:{epoch}]\033[0m loss:{loss.numpy}")



