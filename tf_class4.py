import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# 用RNN 预测character
"""
要求
'a' >> 'b'
'b' >> 'c'
'c' >> 'd'
'd' >> 'e'
'e' >> 'a'
"""
# 1，准备数据集
char2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}

id2char = dict([(v, k) for k, v in char2id.items()])

# 用独热码来编码输入
one_hot = np.diag([1., 1., 1., 1., 1.])

x_train = np.array([one_hot[char2id['a']],
                    one_hot[char2id['b']],
                    one_hot[char2id['c']],
                    one_hot[char2id['d']],
                    one_hot[char2id['e']]])

y_train = np.array([char2id['b'],
                    char2id['c'],
                    char2id['d'],
                    char2id['e'],
                    char2id['a']])



seed = 21
np.random.seed(seed)
np.random.shuffle(x_train)
np.random.seed(seed)
np.random.shuffle(y_train)

# 2, 数据预处理
steps = 1
features = len(x_train[0])

# RNN 中接收 x_train的shape是 （batch，rnn步长，embedding维度）
x_train = np.reshape(x_train, newshape=(len(x_train), steps, features))

# 3, 搭建神经网络框架
hidden_units = 3
model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(hidden_units))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              # 若y是scalar，用SparseCategoricalCrossentropy来计算交叉熵损失
              # 若y被转成one-hot编码，用CategoricalCrossentropy 来计算交叉熵损失
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics = 'sparse_categorical_accuracy')

# 4，训练模型
# 断点续训
ckpt_path = './checkpoint/rnn_character_one_to_one.ckpt'

if os.path.exists(ckpt_path + '.index'):  # + 'index'
    print('load model......')
    model.load_weights(ckpt_path)
    print('finishing loading model')

callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              monitor='loss')

history = model.fit(x_train, y_train,
                    batch_size=16,
                    epochs=500,
                    verbose=1,
                    callbacks=[callback])
model.summary()

"""
注：保存模型到ckpt后，会生成3个文件
checkpoint文件是列出保存的所有模型以及最近模型的相关信息
data文件是包含训练变量的文件；
.index是描述variable中key和value的对应关系；
"""


# 5, 保存weight
weight_file_path = './weights/weight.txt'

if not os.path.exists(weight_file_path):
    os.makedirs(os.path.dirname(weight_file_path), exist_ok=True)

with open(weight_file_path, 'w') as f:
    for w in model.trainable_variables:
        f.write(str(w.name) + '\n')
        f.write(str(w.shape) + '\n')
        f.write(str(w.numpy()) + '\n')


# 6, 分析结构
loss = history.history['loss']
accuray = history.history['sparse_categorical_accuracy']  # 准确率

plt.subplot(1, 2, 1)
plt.plot(loss, label='loss')
plt.title('loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuray, label='accuray')
plt.title('accuray')
plt.legend()
plt.show()


# 7, 用训练好的模型进行预测

x = 'a'
x_test = one_hot[char2id[x]]
x_test = np.reshape(x_test, newshape=(1, 1, len(x_test)))
y_pred = model.predict(x_test)
y_pred = int(tf.argmax(y_pred, axis=1).numpy())
print(x + ' >>>>> ' + id2char[y_pred])


"""
利用embedding 来训练神经网络
embedding_dim = 2
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding([len(char2idx), embedding_dim]))
model.add(tf.keras.layers.SimpleRNN(3))
model.add(tf.keras.layers.Dense(5, activation='softmax'))

"""






