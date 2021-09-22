import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers

# tf2 使用即时执行模式（Eager Execution）， tf1中需要关闭这一模式

# Part 1: tf 基础知识
# 1， 张量的参数

scalar = tf.constant(100)
vector = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
matrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float16)
cube_matrix = tf.constant([[[1], [2], [3], [4], [5], [6], [7], [8], [9]]])

scalar.get_shape()  # 获得张量的形状
vector.get_shape()
print(matrix)    # 在tf2 中，可直接打印tensor获得信息
print(cube_matrix.dtype)  # 获得张量的数据类型结构


# 2，可以通过tf中特殊的函数来生成一个张量

random_float = tf.random.uniform(shape=())
zero_vec = tf.zeros(shape=(2,))
one_matrix = tf.ones(shape=[3, 3])
from_numpy = tf.convert_to_tensor(np.diag([1, 2, 3]))
to_numpy = from_numpy.numpy()  # 将tensor转变成numpy数组
eye = tf.eye(num_rows=2, num_columns=3, batch_shape=(4,))
eye.get_shape()  # [4, 2, 3]

# 3, 张量的计算
A = tf.random.normal(shape=[3, 2])  # 3个样本，每一个样本包含2个特征
B = tf.random.uniform(shape=[3, 2])
C = tf.add(A, B)
B = tf.reshape(B, shape=(2, 3))
D = tf.matmul(A, B)


# 4, tf 变量
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)

# Variables can be all kinds of types, just like tensors
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])

print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy)

# 变量无法重构形状
print("A variable:", my_variable)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.argmax(my_variable))

# This creates a new tensor; it does not reshape the variable.
reshaped_variable = tf.reshape(my_variable, ([1, 4]))  # 是一个tensor不是一个变量
print("\nCopying and reshaping: ", reshaped_variable)
# 重新分配张量
#my_variable.assign(tf.random.normal([2, 3]))  # 形状不一致
# 变量的运算
a = tf.Variable([2.0, 3.0])
# Create b based on the value of a
b = tf.Variable(a)
a.assign([5, 6])

# a and b are different
print(a.numpy())
print(b.numpy())  # a 改变不影响b

c = a+b  # 得到一个tensor，不是variable

# There are other versions of assign
print(a.assign_add([2, 3]).numpy())  # [7. 9.]
print(a.assign_sub([7, 9]).numpy())  # [0. 0.]

# 变量命名
my_var = tf.Variable([2, 3], name='var')
print('my_var:', my_var.name)
my_var_ = tf.Variable([1, 2], name='var')
print('my_var_:', my_var_.name)
print(my_var == my_var_)


# Gradient
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    y = tf.square(x)   # GradientTape上下文内
y_grad = tape.gradient(y, x)  # 求y关于x的梯度
print(y, y_grad)

# 多变量求导
X = tf.constant([[1., 2.], [3., 4.]])  # shape=(2, 2)
w = tf.Variable([[1.], [2.]])  # shape = (2, 1)
b = tf.Variable([[1.], [1.]], trainable=False)  # shape = (2, 1)
y = tf.constant([[1.], [2.]])  # shape = (2, 1)

with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
#w_grad, b_grad = tape.gradient(L, [w, b])
#print(L, w_grad, b_grad)

my_vars = {'w': w,   # 'w'和'b'分别用来表示L关于w和b的梯度，可以取任何名字
           'b': b}

gradients = tape.gradient(L, my_vars)  # 计算梯度，可传递一个变量字典
gradients['b']


#  简单线性回归
"""
使用经典的 Auto MPG 数据集，构建了一个用来预测70年代末到80年代初汽车燃油效率的模型。
1，数据进行预处理
2，构建模型
"""
# 1.1 下载数据集
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

# 使用pandas导入数据
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# 1.2 数据缺失值处理
dataset.isna().sum()  # horsepower列存在na
np.where(dataset.isna() == True)

dataset = dataset.dropna()

# origin 代表产地，"Origin"这个字段需要转换为one-hot编码。
# 将类别数字转化为one-hot编码，以便机器学习到原产地对于MPG的影响。
# 如果不进行one-hot编码，Origin是用1,2,3等数字来表示的，
# 直接这样使用的话机器学习会把Origin当做一个数量而不是类别来对待
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

# 1.3 切分训练数据集和测试数据集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 或者通过更一般的
#from sklearn.model_selection import train_test_split
#train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, test_size=0.2, shuffle=True)

# 1.4 查看训练数据集（seaborn的pairplot提供了图形化的方式快速查看数据集，更高效直观，这里使用pairplot来表现两两的特征对比图。）
# 车子越重，排量越大，气缸越多，燃油的经济性越低(Miles per Galon)
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")  # kde：kernel density estimation
plt.show()

# 1.5 查看训练数据的统计信息
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

# 1.5 从数据中分离标签
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# 1.6 对数据进行标准化
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# 2 构建模型
# 2.1 设置超参数

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 1000
SEED = 30
REGULARIZER_COEFFICIENT = 0.0
HIDDEN_LAYER_SIZE = 64
OUTPUT_SIZE = 1

nbr_of_features = normed_train_data.shape[1]

normed_train_data = tf.cast(normed_train_data, tf.float32)
normed_test_data = tf.cast(normed_test_data, tf.float32)
train_labels = tf.cast(train_labels, tf.float32)
test_labels = tf.cast(test_labels, tf.float32)
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data, train_labels)).batch(BATCH_SIZE)
test_db = tf.data.Dataset.from_tensor_slices((normed_test_data, test_labels)).batch(BATCH_SIZE)

# 2.2 构建一个2层的全连接神经网络
# 1) from scratch


class build_model_from_scratch(object):
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2

    def train(self, x, y):
        h1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        y_pred = tf.matmul(h1, self.w2) + self.b2
        y_pred = tf.squeeze(y_pred)  # 时刻注意 tensor之间的形状
        variables = [self.w1, self.b1, self.w2, self.b2]

        loss = tf.reduce_mean(tf.square(y - y_pred))
        training_MSE = loss
        # 添加正则项
        loss_regularization = []
        loss_regularization.append(tf.nn.l2_loss(self.w1))
        loss_regularization.append(tf.nn.l2_loss(self.w2))
        loss += REGULARIZER_COEFFICIENT * tf.reduce_sum(loss_regularization)
        grads = tape.gradient(loss, variables)
        self.w1.assign_sub(LEARNING_RATE * grads[0])
        self.w2.assign_sub(LEARNING_RATE * grads[2])
        self.b1.assign_sub(LEARNING_RATE * grads[1])
        self.b2.assign_sub(LEARNING_RATE * grads[3])
        return training_MSE


    def predict(self, x_test):
        h1_pred = tf.nn.relu(tf.matmul(x_test, self.w1) + self.b1)
        y_pred = tf.matmul(h1_pred, self.w2) + self.b2
        y_pred = tf.squeeze(y_pred)

        return y_pred  # MSE


# 初始化参数和模型
w1 = tf.Variable(tf.random.normal([nbr_of_features, HIDDEN_LAYER_SIZE]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[HIDDEN_LAYER_SIZE]), dtype=tf.float32)
w2 = tf.Variable(tf.random.truncated_normal([HIDDEN_LAYER_SIZE, OUTPUT_SIZE]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[OUTPUT_SIZE]), dtype=tf.float32)

scratch_model = build_model_from_scratch(w1, b1, w2, b2)

# 训练模型
with tf.GradientTape(persistent=True) as tape:
    for i in range(EPOCHS):
        training_MSE = 0
        training_size = 0
        for (x, y) in train_db:
            training_MSE += scratch_model.train(x, y) * len(y)
            training_size += len(y)
        training_MSE = training_MSE/training_size
        if i % 100 == 0:
            test_MSE = 0
            test_size = 0
            for (x_test, y_test) in test_db:
                y_pred = scratch_model.predict(x_test)
                test_MSE += tf.reduce_sum(tf.square(y_test-y_pred))
                test_size += len(y_test)
            test_MSE = test_MSE/test_size
            print('epoch={}: training error = {} and test error = {}'.format(i, training_MSE, test_MSE))



test_predictions = scratch_model.predict(normed_test_data)

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])



# 利用TF中已有模型构造
def build_model():
    model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(1)])

    optimizer = tf.keras.optimizers.SGD(lr=0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return model

model = build_model()
model.summary()

# 通过为每个完成的时期打印一个点来显示训练进度


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

# 10 epochs 后，若测试集没有改进，则停止训练，
# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels,
                    epochs=EPOCHS, validation_split=0.2, verbose=0,
                    batch_size=BATCH_SIZE,
                    callbacks=[PrintDot()])  # callbacks=[early_stop, PrintDot()]

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

"""
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


plot_history(history)
"""