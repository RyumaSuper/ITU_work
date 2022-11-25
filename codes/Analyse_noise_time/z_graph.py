import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from drawnow import drawnow, figure

def m_slice(path, step, ex):
    
    start = 0
    end = 6622
    files = 5
    p = '/home/ryuma/codes/'
    movie = cv2.VideoCapture(p + path)
    Fs = int(movie.get(cv2.CAP_PROP_FRAME_COUNT)) # フレーム数カウント
    flag, frame = movie.read()
    r_pixels = np.zeros((files, end - start), int) # 調整が必要
    g_pixels = np.zeros((files, end - start), int) # 調整が必要
    b_pixels = np.zeros((files, end - start), int) # 調整が必要
    print(Fs)

    for a in range(0,files): # ここも調整する
        path2 = p + 'received_' + str((20 - 2*a) * 100) + 'kbps_025.mp4'
        movie2 = cv2.VideoCapture(path2)

        path_head = path[:-4] + '_' # str型の名前をpath_headに代入してるだけ
        ext_index = np.arange(0, Fs, step) # 変数stepの間隔で値をext_indexに代入している

        f_time = np.array([])
        r_pixel = np.array([])
        g_pixel = np.array([])
        b_pixel = np.array([])

        for i in range(start, end):
            flag2, frame2 = movie2.read()

            check = i == ext_index # iとext_indexが一致していればcheckがTrue

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

                    cv2.imwrite(os.path.join(p + 'Analyse_noise_time/', path_out), frame) # 指定のパスにフレームを出力(Original)
                    re_path_out = 'received_' + str((20 - 2*a) * 100) + 'kbps_0.25%' + path_out[8:]
                    cv2.imwrite(os.path.join(p + 'Analyse_noise_time/', re_path_out), frame2) # 指定のパスにフレームを出力(Received)
                    f_time = np.append(f_time, i)
                    print(a,i)
                    r_pixel = np.append(r_pixel, r_diff(p, path_out, re_path_out)) # ノイズピクセルを時間軸で数える
                    g_pixel = np.append(g_pixel, g_diff(p, path_out, re_path_out)) # ノイズピク>    セルを時間軸で数える
                    b_pixel = np.append(b_pixel, b_diff(p, path_out, re_path_out)) # ノイズピク>    セルを時間軸で数える
                    print(r_pixel[-1])
                    print(g_pixel[-1])
                    print(b_pixel[-1])
                    #realtime_graph(f_time, w_pixels, w_pixel, a)
                else:
                    pass
            else:
                pass

        r_pixels[a] = r_pixel
        g_pixels[a] = g_pixel
        b_pixels[a] = b_pixel

    graph_pixel(f_time, r_pixels, g_pixels, b_pixels) # グラフ絵画
    return

'''def rgb(path, p): # 今は使ってない
    path = p + path
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.asarray(img)
    path_out_rgb = path[0:-4] + '.txt'

    with open(path_out_rgb, mode='w') as f:
        f.write(str(img_array))
'''

def r_diff(path, o_path, r_path): # ノイズピクセルを時間軸で数える

    # ここら辺で画像を読み込んでnumpy配列と化している
    o_path = path + 'Analyse_noise_time/' + o_path
    r_path = path + 'Analyse_noise_time/' + r_path
    o_img = cv2.imread(o_path)
    r_img = cv2.imread(r_path)
    o_array = np.asarray(o_img)
    r_array = np.asarray(r_img)

    # hとwにはフレームの縦横ピクセル数をnp.shapeで代入
    h, w = o_array.shape[0], o_array.shape[1]

    #d = np.zeros((h, w)) 使ってない

    count = 0
    for i in range(h):
        for j in range(w):
            count += r_array[i][j][2] - o_array[i][j][2]
    return count

def g_diff(path, o_path, r_path): # ノイズピクセルを時間軸で数える

    # ここら辺で画像を読み込んでnumpy配列と化している
    o_path = path + 'Analyse_noise_time/' + o_path
    r_path = path + 'Analyse_noise_time/' + r_path
    o_img = cv2.imread(o_path)
    r_img = cv2.imread(r_path)
    o_array = np.asarray(o_img)
    r_array = np.asarray(r_img)

    # hとwにはフレームの縦横ピクセル数をnp.shapeで代入
    h, w = o_array.shape[0], o_array.shape[1]

    #d = np.zeros((h, w)) 使ってない

    count = 0
    for i in range(h):
        for j in range(w):
           count += r_array[i][j][1] - o_array[i][j][1]
    return count

def b_diff(path, o_path, r_path): # ノイズピクセルを時間軸で数える

    # ここら辺で画像を読み込んでnumpy配列と化している
    o_path = path + 'Analyse_noise_time/' + o_path
    r_path = path + 'Analyse_noise_time/' + r_path
    o_img = cv2.imread(o_path)
    r_img = cv2.imread(r_path)
    o_array = np.asarray(o_img)
    r_array = np.asarray(r_img)

    # hとwにはフレームの縦横ピクセル数をnp.shapeで代入
    h, w = o_array.shape[0], o_array.shape[1]

    #d = np.zeros((h, w)) 使ってない

    count = 0
    for i in range(h):
        for j in range(w):
           count += r_array[i][j][0] - o_array[i][j][0]
    return count
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
    #cv2.imwrite(r_path, r_array) #  塗りつぶしたものを出力
'''
def realtime_graph(f_time, w_pixels, w_pixel, a):
    fig, ax = plt.subplots()
    c1 = "r"
    #c2 = "darkorange"
    #c3 = "gold"
    #c4 = "limegreen"
    #c5 = "turquoise"
    #c6 = "deepskyblue"
    #c7 = "blue"
    c8 = "purple"
    #c9 = "magenta"
    #c10 = "deeppink"
    l1 = "1100kbps"
    #l2 = "1200kbps"
    #l3 = "1300kbps"
    #l4 = "1400kbps"
    #l5 = "1500kbps"
    #l6 = "1600kbps"
    #l7 = "1700kbps"
    l8 = "2000kbps"
    #l9 = "1900kbps"
    #l10 = "2000kbps"
    
    ax.set_xlabel('time')
    ax.set_ylabel('noise pixels')
    ax.set_title('noise pixels graph')
    ax.grid()
    if a >= 1100:
        line1100, = ax.plot(f_time, w_pixel, color=c1, label=l1)
        line1100.set_data(f_time, w_pixel)
        ax.set_xlim((f_time.min(), f_time.max()))
        ax.set_ylim((w_pixel.min(), w_pixel.max()))
    elif a >= 1200:    
        line1100, = ax.plot(f_time, w_pixels[0,:], color=c1, label=l1)
        line1200, = ax.plot(f_time, w_pixel, color=c2, label=l2)
        line1100.set_data(f_time, w_pixels[0,:])
        line1200.set_data(f_time, w_pixel)
    elif a >= 1300:
        line1100, = ax.plot(f_time, w_pixels[0,:], color=c1, label=l1)
        line1200, = ax.plot(f_time, w_pixels[1,:], color=c2, label=l2)
        line1300, = ax.plot(f_time, w_pixel, color=c3, label=l3)
        line1100.set_data(f_time, w_pixels[0,:])
        line1200.set_data(f_time, w_pixels[1,:])
        line1300.set_data(f_time, w_pixel)
    elif a >= 1400:
        line1100, = ax.plot(f_time, w_pixels[0,:], color=c1, label=l1)
        line1200, = ax.plot(f_time, w_pixels[1,:], color=c2, label=l2)
        line1300, = ax.plot(f_time, w_pixels[2,:], color=c3, label=l3)
        line1400, = ax.plot(f_time, w_pixels, color=c4, label=l4)
        line1100.set_data(f_time, w_pixels[0,:])
        line1200.set_data(f_time, w_pixels[1,:])
        line1300.set_data(f_time, w_pixels[2,:])
        line1400.set_data(f_time, w_pixel)
    else:
        pass


    ax.legend(loc=0)
    fig.tight_layout()
    plt.pause(0.01)
   ''' 

def graph_pixel(f_time, r_pixels, g_pixels, b_pixels):
    fig, ax = plt.subplots()
    c1 = "blue"
    c2 = "forestgreen"
    c3 = "gold"
    c4 = "red"
    #c5 = "purple"
    #c6 = "deepskyblue"
    #c7 = "blue"
    #c8 = "blue"
    #c9 = "magenta"
    #c10 = "deeppink"

    l1 = "2000-1800kbps"
    l2 = "2000-1600kbps"
    l3 = "2000-1400kbps"
    l4 = "2000-1200kbps"
    #l5 = "1200kbps"
    #l6 = "2000kbps blue"
    #l7 = "1700kbps"
    #l8 = "2000kbps"
    #l9 = "1900kbps"
    #l10 = "2000kbps"

    ax.set_xlabel('time')
    ax.set_ylabel('noise pixels')
    ax.set_title('noise pixels graph')
    ax.grid()

    ave20 = (r_pixels[0,:] + g_pixels[0,:] + b_pixels[0,:]) / 3
    ave18 = (r_pixels[1,:] + g_pixels[1,:] + b_pixels[1,:]) / 3
    ave16 = (r_pixels[2,:] + g_pixels[2,:] + b_pixels[2,:]) / 3
    ave14 = (r_pixels[3,:] + g_pixels[3,:] + b_pixels[3,:]) / 3
    ave12 = (r_pixels[4,:] + g_pixels[4,:] + b_pixels[4,:]) / 3
    ax.plot(f_time, ave20 - ave18, color=c1, label=l1)
    ax.plot(f_time, ave20 - ave16, color=c2, label=l2)
    ax.plot(f_time, ave20 - ave14, color=c3, label=l3)
    ax.plot(f_time, ave20 - ave12, color=c4, label=l4)
    #ax.plot(f_time, ave12, color=c5, label=l5)
    #ax.plot(f_time, b_pixels[1,:], color=c6, label=l6)
    #ax.plot(f_time, w_pixels[6,:], color=c7, label=l7)
    #ax.plot(f_time, w_pixels[1,:], color=c8, label=l8)
    #ax.plot(f_time, w_pixels[8,:], color=c9, label=l9)
    #ax.plot(f_time, w_pixels[9,:], color=c10, label=l10)

    ax.legend(loc=0)
    fig.tight_layout()
    plt.savefig('Data_differences.png')
    plt.show()


m_slice('original.mp4', 1, '.png') # プログラムの実行
