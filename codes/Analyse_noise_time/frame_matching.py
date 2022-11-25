import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img1 = cv.imread('/Users/ryuma/Desktop/コンテスト関連/ITU/codes/Matching_frames/origin_frame_03500.png')
img2 = cv.imread('/Users/ryuma/Desktop/コンテスト関連/ITU/codes/Matching_frames/received_frame_03500.png')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

#print('step1')
#gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
#gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

print('step2')
orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print('step3')
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

print('step4')
matches = sorted(matches, key = lambda x:x.distance)
dst = cv.drawMatches(img1, kp1, img2, kp2, matches[:500], None, flags=2)

print('step5')
#cv.imshow('dst', dst)
#cv.waitKey(0)
plt.imshow(dst),plt.show()
