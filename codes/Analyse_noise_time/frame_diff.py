import cv2
import numpy as np
import os

def m_slice(path, path2, dir, step, ex):
    movie = cv2.VideoCapture(path) # この2行でパスの動画を読み込む OriginalとReceivedそれぞれ
    movie2 = cv2.VideoCapture(path2)

    Fs = int(movie.get(cv2.CAP_PROP_FRAME_COUNT)) # フレーム数カウント
    
    path_head = dir + '_frame_' # str型の名前をpath_headに代入してるだけ
    ext_index = np.arange(0, Fs, step) # 変数stepの間隔で値をext_indexに代入している

    for i in range(Fs - 1):
        flag, frame = movie.read() # この2行で動画のフレームを読み込み
        flag2, frame2 = movie2.read()

        check = i == ext_index # iとext_indexが一致していればcheckがTrue
        p = '/Users/ryuma/Desktop/コンテスト関連/ITU/codes/Green_frames/' # ファイル出力するときのパス
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
                re_path_out = 'received' + path_out[6:]
                cv2.imwrite(os.path.join(p, re_path_out), frame2) # 指定のパスにフレームを出力(Received)
                #rgb(path_out, p)
                #rgb(re_path_out, p)
                analyse_rgb(p, path_out, re_path_out) # 緑色に塗りつぶす関数
            else:
                pass
        else:
            pass
    return

def rgb(path, p): # 今は使ってない
    path = p + path
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.asarray(img)
    path_out_rgb = path[0:-4] + '.txt'

    with open(path_out_rgb, mode='w') as f:
        f.write(str(img_array))

def analyse_rgb(path, o_path, r_path): # 緑色に塗りつぶす関数

    # ここら辺で画像を読み込んでnumpy配列と化している
    o_path = path + o_path
    r_path = path + r_path
    o_img = cv2.imread(o_path)
    r_img = cv2.imread(r_path)
    o_array = np.asarray(o_img)
    r_array = np.asarray(r_img)

    # hとwにはフレームの縦横ピクセル数をnp.shapeで代入
    h, w = o_array.shape[0], o_array.shape[1]

    #d = np.zeros((h, w)) 使ってない


    # ここで実際に塗りつぶしている。
    # RGBそれぞれの値がOriginalより50以上離れていなかったら緑に塗りつぶさない
    # RGBどれか1つでもOriginalより50以上離れていたらそのピクセルを緑にする
    for i in range(h):
        for j in range(w):
            c = np.isclose(o_array[i][j], r_array[i][j], rtol=0, atol=50)
            if c[0] == True and c[1] == True and c[2] == True:
                pass
                #d[i][j] = 1
            else:
                #d[i][j] = 0
                r_array[i][j][0] = r_array[i][j][2] = 0
                r_array[i][j][1] = 255
    '''Falsebit = 0

    left = right = w
    top = bottom = h
    first = -1
    for i in range(h):
        for j in range(w):
            if d[i][j] == Falsebit and first == -1:
                top = bottom = i
                left = right = j
                first = 1
            if d[i][j] == Falsebit and left >= j:
                left = j
            if d[i][j] == Falsebit and right <= j:
                right = j
            if d[i][j] == Falsebit and bottom <= i:
                bottom = i
    cv2.rectangle(r_array, (left, top), (right, bottom), (0, 255, 0), thickness=8, lineType=cv2.LINE_4)'''
    cv2.imwrite(r_path, r_array) #  塗りつぶしたものを出力

m_slice('original.mp4', 'received.mp4', 'origin', 500, '.png') # プログラムの実行
