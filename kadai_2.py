import numpy as np
import cv2


threshold = 110 #二値化閾値変数

#画像読み込み
original = cv2.imread('metal_panel.jpg')

#画像サイズの取得
ho, wo = original.shape[:2]
#画像をリサイズ（パソコンの都合上）
bgr = cv2.resize(original,(wo//5, ho//5))
#リサイズ後のサイズ取得
h,w = bgr.shape[:2]

#----前景抽出処理----
mask = np.zeros((h,w), dtype = np.uint8)
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
rect = (1,1,w, h)

#前景抽出
cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
bgr2 = bgr*mask2[:,:,np.newaxis]

#グレースケール変換
img_gray = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)

#平均化処理
img_ave = cv2.blur(img_gray,(10,10))

#----2値化処理----
ret, img_thresh = cv2.threshold(img_ave, threshold, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = list(filter(lambda x: cv2.contourArea(x) > 10000, contours))
mask5 = np.zeros_like(bgr)
cv2.drawContours(mask5, contours,-1, (255,255,255), thickness = -1)
img_masked = cv2.bitwise_and(bgr, mask5)

#画像表示
cv2.imshow('original', bgr)
cv2.imshow('masked', img_masked)

#後処理
cv2.waitKey(0)
cv2.destroyAllWindows()
