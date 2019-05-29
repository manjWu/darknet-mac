import cv2
import numpy as np


# 当调节滑块时，调用这个函数。这个没有使用到、】【asdfgh
def do_nothing(x):
    pass


# 创建黑图像
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')

# 创建滑块,注册回调函数
cv2.createTrackbar('R', 'image', 0, 255, do_nothing)
cv2.createTrackbar('G', 'image', 0, 255, do_nothing)
cv2.createTrackbar('B', 'image', 0, 255, do_nothing)

while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # 获得滑块的位置
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')

    # 设置图像颜色
    img[:] = [b, g, r]

cv2.destroyAllWindows()