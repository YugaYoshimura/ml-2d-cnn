#　必要なモジュールをimportする
import numpy as np
import random
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import codecs as cd
import os

from const_pai import code2disphai, code2hai

# TensorFlowのログレベルを変更
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = 全て表示, 1 = INFOを非表示, 2 = WARNINGを非表示, 3 = ERRORを非表示


# 多クラス分類として学習する場合
CATEGORICAL = True


# ここでdata.pklを読み込む
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# ここでdata.pklの中身を確認する
# print(data[0][0])

# データの読み込み
# データをXとYに分ける
X = [item[0] for item in data]
#print(X[0])
Y = [item[1] for item in data]
#print(Y[0])

"""
X_orig = []
Y_orig = []
if CATEGORICAL:
  X, Y = normalize_class_data(X_orig, Y_orig)
else:
  X, Y = normalize_data(X_orig, Y_orig)
"""

# データをランダムにシャッフル
idx_list = list(range(len(X)))
random.shuffle(idx_list)

"""
X_orig_shuffled = [X_orig[idx] for idx in idx_list]
Y_orig_shuffled = [Y_orig[idx] for idx in idx_list]
"""

X_shuffled = [X[idx] for idx in idx_list]
Y_shuffled = [Y[idx] for idx in idx_list]


# シャッフルされたデータを使うように変更
split_index = int(len(X_shuffled)*0.8)

"""
train_x_orig = X_orig_shuffled[:split_index]
train_y_orig = Y_orig_shuffled[:split_index]
test_x_orig = X_orig_shuffled[split_index:]
test_y_orig = Y_orig_shuffled[split_index:]
"""

train_x = X_shuffled[:split_index]
train_y = Y_shuffled[:split_index]
test_x = X_shuffled[split_index:]
test_y = Y_shuffled[split_index:]
print(len(train_x))

# ラベル値をチェックして範囲外の値があればエラーメッセージを表示
valid_labels = list(range(len(code2hai)))
for y in train_y + test_y:
    if y not in valid_labels:
        raise ValueError(f"Received a label value of {y} which is outside the valid range of [0, {len(code2hai) - 1}).")


# 学習モデル作り
train_x = np.array(train_x)
train_x = train_x.reshape(train_x.shape[0], -1, 1)
test_x = np.array(test_x)
test_x = test_x.reshape(test_x.shape[0], -1, 1)

train_y = np.array(train_y)
test_y = np.array(test_y)


input_shape = train_x.shape[1:]
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(32, 3, activation='relu',input_shape=input_shape))
model.add(tf.keras.layers.Dense(16,activation='relu'))
model.add(tf.keras.layers.Flatten())
if CATEGORICAL:
  model.add(tf.keras.layers.Dense(len(code2hai), activation="softmax"))
  model.compile(loss = "sparse_categorical_crossentropy", optimizer="adam")
else:
  model.add(tf.keras.layers.Dense(1))
  model.compile(loss = "mse", optimizer="adam")


# 学習前のモデルで予測したものをプロットしてみる
if CATEGORICAL:
  plt.plot(np.arange(len(test_y)), test_y, label="dahai")
  predict_y = model.predict(test_x)
  predict_y = np.argmax(predict_y, axis=-1)
  # predict_y = predict_y.reshape(predict_y.shape[:2])
  plt.plot(np.arange(len(test_y)), predict_y, label="predict(before)")
  plt.legend()
else:
  plt.plot(np.arange(len(test_y)), test_y, label="dahai")
  predict_y = model.predict(test_x)
  predict_y = predict_y.reshape(predict_y.shape[:2])
  plt.plot(np.arange(len(test_y)), predict_y, label="predict(before)")
  plt.legend()
     

# 学習させる
hist = model.fit(train_x, train_y, batch_size=16, epochs=100, verbose=1)
history = hist.history
plt.plot(hist.epoch, history["loss"], label="loss")
plt.show()


#学習後の予測をプロット
if CATEGORICAL:
  plt.plot(np.arange(len(test_y)), test_y, label="dahai")
  predict_y = model.predict(test_x)
  predict_y = np.argmax(predict_y, axis=-1)
  # predict_y = predict_y.reshape(predict_y.shape[:2])
  plt.plot(np.arange(len(test_y)), predict_y, label="predict(after)")
  plt.legend()
else:
  plt.plot(np.arange(len(test_y)), test_y, label="dahai")
  predict_y = model.predict(test_x)
  predict_y = predict_y.reshape(predict_y.shape[:2])
  plt.plot(np.arange(len(test_y)), predict_y, label="predict(after)")
  plt.legend()


