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
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)




