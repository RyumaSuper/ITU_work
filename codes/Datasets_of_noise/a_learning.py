# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
import os
import subprocess

# --------------------ここから俺が追加したところ-----------------------------
path = '/home/ryuma/codes/Datasets_of_noise/' # データセットを取るためのパス

h = 90
w = 160

c = 0
c_img = np.array([]) # ノイズじゃない画像をいれる配列
n_img = np.array([]) # ノイズ画像をいれる配列


# ノイズじゃない画像を数えて、配列にぶっ込んでる
while True:
    if os.path.exists(path + 'clear_' + str(c) + '.png') == False:
        break
    c_img = np.append(c_img, cv2.imread('clear_' + str(c)+".png"))

    c += 1
print('clear_image:' + str(c))
c_img = c_img.reshape(c, h, w, 3) # 配列の形を変形



n = 0
# ここでもノイズ画像を数えて配列にぶっ込んでる
while True:
    if os.path.exists(path + 'noise_' + str(n) + '.png') == False:
        break
    n_img = np.append(n_img, cv2.imread('noise_' + str(n)+".png"))
    n += 1

print('noise_image:' + str(n))
n_img = n_img.reshape(n, h, w, 3) # 配列を変形

# x_img = np.array([])
x_img = np.concatenate([c_img, n_img], 0) # ノイズ画像とそうじゃない画像を1つにまとめている

# ここら辺でラベル付けしている.ノイズ画像は0,そうじゃないのは1
y_img = np.full((c,), 1)
add_img = np.full((n,), 0)
y_img = np.append(y_img, add_img)
# ---------------------------------------------ここまで-------------------------------------

# keras用のパラメータ
batch_size = 128
epochs = 50

# 数字画像のサイズ 縦(row)と横(col)
img_rows, img_cols = h, w # 縦横サイズはデータセットの画像をみて適宜変更する必要がある

# 学習結果を保存するファイルの決定
if len(sys.argv)==1:
    print('使用法: python3 a_learning.py 保存ファイル名.h5')
    sys.exit()
savefile = sys.argv[1]

# 特徴量のセットを変数Xに、ターゲットを変数yに格納
X = x_img
y = y_img

# クラス数の取り出し
n_classes = len(np.unique(y))

# データXをCNN用の形式に変換
if K.image_data_format() == 'channels_first':
    print('channels first')
    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X = X.reshape(X.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
# ターゲットyをkeras用の形式に変換
y_keras = keras.utils.to_categorical(y, n_classes)

# 畳み込みニューラルネットワークを定義
#model = Sequential()
#model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(units=128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(units=n_classes, activation='softmax'))

# 畳み込みニューラルネットワークを定義 Ryuma
model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(7, 7), activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=16, kernel_size=(7, 7), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=n_classes, activation='softmax'))

# モデルのコンパイル
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# モデルの学習
history = model.fit(X, y_keras, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)

# 結果の表示
result = model.predict_classes(X, verbose=0)

# データ数をtotalに格納
total = len(X)
# ターゲット（正解）と予測が一致した数をsuccessに格納
success = sum(result==y)

# 正解率をパーセント表示
print('正解率')
print(100.0*success/total)

# 学習結果を保存
model.save(savefile)

# 損失関数のグラフの軸ラベルを設定
plt.xlabel('time step')
plt.ylabel('loss')

# グラフ縦軸の範囲を0以上と定める
plt.ylim(0, max(np.r_[history.history['val_loss'], history.history['loss']]))

# 損失関数の時間変化を描画
val_loss, = plt.plot(history.history['val_loss'], c='#56B4E9')
loss, = plt.plot(history.history['loss'], c='#E69F00')

# グラフの凡例（はんれい）を追加
plt.legend([loss, val_loss], ['loss', 'val_loss'])

# 絵画した画像の保存
plt.savefig("a_lossgraph.png")
