#　必要なモジュールをimportする
import numpy as np
import random
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import codecs as cd
import os
import argparse
import sys
from const_pai import code2disphai, code2hai

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # 追加

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_folder", type=str, help="folder to save results.")
    return parser.parse_args()

args = parse_arg()
save_folder = args.save_folder

# TensorFlowのログレベルを変更pip install scikit-learnpip install scikit-learnpip install scikit-learn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = 全て表示, 1 = INFOを非表示, 2 = WARNINGを非表示, 3 = ERRORを非表示


# 多クラス分類として学習する場合
CATEGORICAL = True


# ここでdata.pklを読み込む
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# データをXとYに分ける
X = [item[0] for item in data]
Y = [item[1] for item in data]

# ここでX,Yを二次元に変換する
X = np.array(X)
Y = np.array(Y)

# データの形状を確認
#print(f"Original X shape: {X.shape}")
#print(f"Original Y shape: {Y.shape}")

# 各要素を二次元にリシェイプ
X_pad = np.zeros((X.shape[0], 15))
X = np.append(X, X_pad, axis=1)
X = X.reshape(X.shape[0], 22, 23, 1)

# リシェイプ後のデータの形状を確認
#print(f"Reshaped X shape: {X.shape}")
#print(f"Reshaped Y shape: {Y.shape}")

# データをランダムにシャッフル
idx_list = np.arange(len(X))
np.random.shuffle(idx_list)
X_shuffled = X[idx_list]
Y_shuffled = Y[idx_list]

# シャッフルされたデータを使うように変更
split_index = int(len(X_shuffled)*0.8)

train_x = X_shuffled[:split_index]
train_y = Y_shuffled[:split_index]
test_x = X_shuffled[split_index:]
test_y = Y_shuffled[split_index:]

# データを4次元にリシェイプ (例: (サンプル数, 高さ, 幅, チャンネル数))
#train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1, 1)
#test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1, 1)

# リシェイプ後のデータの形状を確認
#print(f"Reshaped X shape: {X.shape}")

# データを訓練データとテストデータに分割
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)

# データの前処理
#scaler = StandardScaler()  # scalerの定義
#train_x = train_x.reshape(train_x.shape[0], -1)  # 2次元にリシェイプしてスケーリング
#test_x = test_x.reshape(test_x.shape[0], -1)  # 2次元にリシェイプしてスケーリング
#train_x = scaler.fit_transform(train_x)
#test_x = scaler.transform(test_x)
#train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1, 1)  # 元の形状に戻す
#test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1, 1)  # 元の形状に戻すに戻す
#test_x = test_x.reshape(test_x.shape[0], X.shape[1], 1, 1)  # 元の形状に戻す

# 学習モデル作り
input_shape = train_x.shape[1:]
print(input_shape)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
if CATEGORICAL:
    model.add(tf.keras.layers.Dense(len(code2hai), activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
else:
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mse", optimizer="adam")

# 学習させる
hist = model.fit(train_x, train_y, batch_size=16, epochs=1, verbose=1)
history = hist.history
plt.figure(figsize=(10, 6))
plt.scatter(hist.epoch, history["loss"], label="loss")
plt.legend()
plt.savefig(os.path.join(save_folder, "loss.png"))  # プロットを保存
#plt.show()

# 学習後の予測をプロット
if CATEGORICAL:
    plt.figure(figsize=(10, 6))
    indices = np.random.choice(len(test_y), size=100, replace=False)  # ランダムに100個サンプリング
    plt.scatter(indices, test_y[indices], label="dahai")
    predict_y = model.predict(test_x)
    predict_y = np.argmax(predict_y, axis=-1)
    plt.scatter(indices, predict_y[indices], label="predict(after)")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "categorical_predictions_after.png"))  # カテゴリカル予測のプロットを保存
else:
    plt.figure(figsize=(10, 6))
    indices = np.random.choice(len(test_y), size=100, replace=False)  # ランダムに100個サンプリング
    plt.scatter(indices, test_y[indices], label="dahai")
    predict_y = model.predict(test_x)
    predict_y = predict_y.reshape(predict_y.shape[:2])
    plt.scatter(indices, predict_y[indices], label="predict(after)")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "regression_predictions_after.png"))  # 回帰予測のプロットを保存

#plt.show()

# 予測結果のヒストグラムをプロット
plt.figure(figsize=(10, 6))
plt.hist(predict_y, bins=50, alpha=0.7, label="predict(after)")
plt.legend()
plt.savefig(os.path.join(save_folder, "predictions_histogram.png"))  # ヒストグラムを保存
#plt.show()

# 予測結果のボックスプロットをプロット
plt.figure(figsize=(10, 6))
plt.boxplot(predict_y, vert=False)
plt.title("Predictions Boxplot")
plt.savefig(os.path.join(save_folder, "predictions_boxplot.png"))  # ボックスプロットを保存
#plt.show()


