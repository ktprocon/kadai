import numpy as np
import cv2


threshold = 137 #二値化閾値変数

#画像読み込み
img_original = cv2.imread('milkdrop.bmp')
#画像ウィンドウ表示
cv2.imshow('original', img_original)

#グレースケールへ変更
img_gray = cv2.cvtColor(img_original,cv2.COLOR_BGR2GRAY)

#二値化処理
ret, img_thresh = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)

#輪郭抽出
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#輪郭データをエリアサイズで選別
contours = list(filter(lambda x: cv2.contourArea(x) > 2000, contours))

#マスク作成
#黒塗り画像作成
mask = np.zeros_like(img_original)
#輪郭データから範囲内の画像を白抜き
cv2.drawContours(mask, contours,-1, (255,255,255), thickness = -1)

#マスク画像で切り抜き
img_masked = cv2.bitwise_and(img_original, mask)

#切り抜き画像を表示
cv2.imshow('masked', img_masked)

#後処理
cv2.waitKey(0)
cv2.destroyAllWindows()