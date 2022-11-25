import cv2
import numpy as np
import os
from PIL import Image
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img

#region_o = list()
#region_r = list()

def m_slice(path, path2, dir, step, ex):
    movie = cv2.VideoCapture(path) # この2行でパスの動画を読み込む OriginalとReceivedそれぞれ
    lst_results = np.array([])
    for a in range(1100, 2001, 100):
        
        region_o = list()
        region_r = list()

        movie2 = cv2.VideoCapture(path2 + str(a) + 'kbps_025.mp4')

        Fs = int(movie.get(cv2.CAP_PROP_FRAME_COUNT)) # フレーム数カウント
    
        path_head = dir + '_'
        ext_index = np.arange(0, Fs, step) # 変数stepの間隔で値をext_indexに代入している

        for i in range(Fs - 1):
            flag, frame = movie.read() # この2行で動画のフレームを読み込み
            flag2, frame2 = movie2.read()

            check = i == ext_index # iとext_indexが一致していればcheckがTrue
            p = '/home/ryuma/codes/Prediction/' # ファイル出力するときのパス
            if flag2 == True:
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
                    if a == 1100:
                        cv2.imwrite(os.path.join(p, path_out), frame) # 指定のパスにフレームを出力(Original)
                    re_path_out = 'received_' + str(a) + 'kbps' + path_out[8:]
                    cv2.imwrite(os.path.join(p, re_path_out), frame2) # 指定のパスにフレームを出力(Received)
                    read_frame(p, path_out, re_path_out, region_o, region_r) # ノイズ画像とそうじゃない画像のデータセットを作る関数
                else:
                    pass
            else:
                pass

        for num in range(len(region_o)):
            if a == 1100:
                region_o[num].save('slice_original_' + str(num) + '.png', "PNG")
            region_r[num].save('slice_received_' + str(a) + 'kbps_' + str(num) + '.png', "PNG")

        c = 0
        while True:
            if os.path.exists(p + 'slice_received_' + str(a) + 'kbps_' + str(c) + '.png') == False:
                break
            c += 1
        print('Number of images:' + str(c))
        lst_results = np.append(lst_results, predict(a, c, lst_results))
    print(lst_results)
    
    return

def read_frame(path, o_path, r_path, region_o, region_r): # データセットを作る関数
   
    sp = 8 # 縦横それぞれいくつに分割するか.デフォルトは8

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
    h, w = r_array.shape[0], r_array.shape[1]
    hsp = int(h / sp)
    wsp = int(w / sp)

    print(r_path)
    # ここでノイズピクセルをカウントして、ノイズ画像とそうじゃないので画像を分割して分けている
    for i in range(sp):
        hmin = int(hsp * i)
        hmax = int(hsp * (i + 1))

        for j in range(sp):
            wmin = int(wsp * j)
            wmax = int(wsp * (j + 1))
            box = (wmin, hmin, wmax, hmax)
            region_o.append(o_im.crop(box))
            region_r.append(r_im.crop(box))

def predict(a, c, lst_results):
    model = load_model('estimate4.h5')
    h = 90
    w = 160
    ori_noise = 0
    ori_noise_pre = 0
    rec_noise = 0
    rec_noise_pre = 0
    for i in range(c):
        print(i,'/',c)
        o_png = 'slice_original_' + str(i) + '.png'
        r_png = 'slice_received_' + str(a) + 'kbps_' + str(i) + '.png'

        o_image = cv2.imread(o_png) / 255
        r_image = cv2.imread(r_png) / 255

        o_image = o_image.reshape(h,w,3)
        r_image = r_image.reshape(h,w,3)
        o_nad = o_image[None, ...]
        r_nad = r_image[None, ...]

        label = ['noise', 'clear']
        o_pre = model.predict(o_nad, batch_size=1, verbose=0)
        r_pre = model.predict(r_nad, batch_size=1, verbose=0)

        o_score = np.max(o_pre)
        r_score = np.max(r_pre)

        o_pre_label = label[np.argmax(o_pre[0])]
        r_pre_label = label[np.argmax(r_pre[0])]

        print('original name:', o_pre_label)
        print('original score:', o_score)
        ori_noise_pre += o_score
        print('original score rate:', ori_noise_pre / (i+1))
        if o_pre_label == 'noise':
            ori_noise += 1
        print('received name:', r_pre_label)
        print('received score:',r_score)
        rec_noise_pre += r_score
        print('received score rate:', rec_noise_pre / (i+1))
        if r_pre_label == 'noise':
            rec_noise += 1
        print(a, 'kbps')
        print('lst_results:', lst_results)

    print('original noise:', ori_noise)
    print('original score rate:', ori_noise_pre / c)
    print('received noise:', rec_noise)
    print('received score rate:', rec_noise_pre / c)
    print('difference:', (ori_noise_pre / c) - (rec_noise_pre / c))
    return ((ori_noise_pre / c) - (rec_noise_pre / c)) * 100

original = '/home/ryuma/codes/original.mp4' # オリジナル動画のパス,自分で変更してね
received = '/home/ryuma/codes/received_' # レシーブ動画のパス,変更してね
m_slice(original, received, 'original', 10, '.png') # プログラムの実行
