import cv2
import time

video = '/Volumes/MachsionHD/drone_实验视频/实验原始视频（部分转换格式）/20161221_教九背后2.mp4'
cv2.namedWindow('video')
cap = cv2.VideoCapture(video)
# 当前滑块的位置
current_pos = 0
# 获得视频总帧数
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame:", frames)


# 在轨迹条位置改变的时候来调用回调函数
# 滑块位置主动: 移动滑块，读取滑块位置，再读取视频
def onTrackbarslide(pos):
    # 设置视频帧的读取位置
    print("onTrackbarslide...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)


# 创建滑块
cv2.createTrackbar('time', 'video', 0, frames, onTrackbarslide)

current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

cv2.setTrackbarPos('time', 'video', int(current_pos))

cv2.namedWindow('result')

while 1:

    # 滑块位置随动: 读视频帧数，再移动滑块
    ret, frame = cap.read()
    # 获取当前视频帧数
    #	current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    #	print("current_pos:",current_pos)
    # 将滑块移动到当前帧
    img = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
    cv2.imshow("result", img)
    tic = time.time()
    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    print(time.time() - tic)
