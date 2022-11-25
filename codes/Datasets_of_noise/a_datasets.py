import cv2
import numpy as np
import os
from PIL import Image
import subprocess
from subprocess import PIPE

region_n = list() # ノイズ画像をいれるための配列
region_c = list() # ノイズじゃない画像をいれるための配列

def m_slice(path, path2, dir, step, ex):
    movie = cv2.VideoCapture(path) # この2行でパスの動画を読み込む OriginalとReceivedそれぞれ
    movie2 = cv2.VideoCapture(path2)

    Fs = int(movie.get(cv2.CAP_PROP_FRAME_COUNT)) # フレーム数カウント
    
    path_head = dir + '_'
    ext_index = np.arange(0, Fs, step) # 変数stepの間隔で値をext_indexに代入している

    for i in range(Fs - 1):
        flag, frame = movie.read() # この2行で動画のフレームを読み込み
        flag2, frame2 = movie2.read()

        check = i == ext_index # iとext_indexが一致していればcheckがTrue
        p = '/home/ryuma/codes/Datasets_of_noise/' # ファイル出力するときのパス
        if flag == True:
            if True in check: # ここら辺は出力するフレームの数字が綺麗になるように処理しているだけ
                if i < 10:
                    path_out = path_head + '0000' + str(i) + ex
                elif i < 100:
                    path_out = path_head + '000' + str(i) + ex
                elif i < 1000:
                    path_out = path_head + '00' + str(i) + ex
                elif i < 10000:
                    path_out = path_head + '0' + str(i) + ex
                else:
                    path_out = path_head + str(i) + ex

                cv2.imwrite(os.path.join(p, path_out), frame) # 指定のパスにフレームを出力(Original)
                re_path_out = 'received' + path_out[8:]
                cv2.imwrite(os.path.join(p, re_path_out), frame2) # 指定のパスにフレームを出力(Received)
                analyse_rgb(p, path_out, re_path_out) # ノイズ画像とそうじゃない画像のデータセットを作る関数
            else:
                pass
        else:
            pass
    # ここら辺で、ノイズ画像とそうじゃない画像に番号割り振って出力している
    for num in range(len(region_n)):
        region_n[num].save('noise_' + str(num) + '.png', "PNG")
    for i in range(len(region_c)):
        region_c[i].save('clear_' + str(i) + '.png', "PNG")
    
    return

def analyse_rgb(path, o_path, r_path): # データセットを作る関数
    
    sp = 8 # 縦横それぞれいくつに画像を分割するか.

    # ここら辺で画像を読み込んでnumpy配列と化している
    o_path = path + o_path
    r_path = path + r_path
    o_img = cv2.imread(o_path)
    r_img = cv2.imread(r_path)
    o_array = np.asarray(o_img)
    r_array = np.asarray(r_img)
    
    o_im = Image.open(o_path)
    r_im = Image.open(r_path)

    # hとwにはフレームの縦横ピクセル数をnp.shapeで代入
    h, w = o_array.shape[0], o_array.shape[1]
    hsp = int(h / sp)
    wsp = int(w / sp)

    #d = np.zeros((h, w)) # ノイズピクセルをいれるための配列
    r = 0
    g = 0
    b = 0
    # ノイズピクセルには0 そうじゃないピクセルには1を入れている
    '''for i in range(h):
        for j in range(w):
            r += o_array[i][j][0]
            g += o_array[i][j][1]
            b += o_array[i][j][2]
            rr += r_array[i][j][0]
            gg += r_array[i][j][1]
            bb += r_array[i][j][2]
    
    print(o_path)
    print(r, rr)
    print(g, gg)
    print(b, bb)
    '''

    c = 0 # ノイズとしてとった画像数を格納する変数
    rate = int(255 *hsp*wsp / 4)
    mrate = int(255 *hsp*wsp / 32)
    
    print(o_path)
    
    for i in range(sp):
        hmin = int(hsp * i)
        hmax = int(hsp * (i + 1))

        for j in range(sp):
            wmin = int(wsp * j)
            wmax = int(wsp * (j + 1))

            for k in range(hmin, hmax):
                for l in range(wmin, wmax):
                    r += int(o_array[k][l][2]) - int(r_array[k][l][2]) # Rの値を数えている
                    g += int(o_array[k][l][1]) - int(r_array[k][l][1]) # Gを数える
                    b += int(o_array[k][l][0]) - int(r_array[k][l][0]) # Bを数える
            box = (wmin, hmin, wmax, hmax)
            if j % 12 == 0:
                region_c.append(o_im.crop(box))
            if (r >= rate or -1*r >= rate) or (g >= rate or -1*g >= rate) or (b >= rate or -1*b >= rate):
                region_n.append(r_im.crop(box))
                c += 1
            r = 0 # Rの初期化
            g = 0 # Gの初期化
            b = 0 # Bの初期化
    print('noise rate', c, '/', sp*sp)

original = '/home/ryuma/codes/original.mp4' # オリジナル動画のパス,自分で変更してね
received = '/home/ryuma/codes/received_1100kbps_025.mp4' # レシーブ動画のパス,変更してね
m_slice(original, received, 'original', 100, '.png') # プログラムの実行
