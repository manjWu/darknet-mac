import cv2
import time

video = '/Volumes/MachsionHD/drone_实验视频/实验原始视频（部分转换格式）/20190403紫金港/20190403紫金港112m-100m-实验1/20190403紫金港全景112m-100m.mp4'
cv2.namedWindow('video')
cv2.namedWindow('result')
cap = cv2.VideoCapture(video)
# 当前滑块的位置
current_pos = 0
# 获得视频总帧数
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame:",frames)

# 在轨迹条位置改变的时候来调用回调函数
# 滑块位置主动: 移动滑块，读取滑块位置，再读取视频
def onTrackbarslide(pos):
	# 设置视频帧的读取位置
	print("onTrackbarslide...")
	cap.set(cv2.CAP_PROP_POS_FRAMES,pos)


# 创建滑块
cv2.createTrackbar('time', 'video', 0, frames, onTrackbarslide)


while(cap.isOpened()):
	time1 = time.time()
	#滑块位置随动: 读视频帧数，再移动滑块
	ret, frame = cap.read()
	print(frame.shape)
	# 获取当前视频帧数
	current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
	print("current_pos:",current_pos)
	# 将滑块移动到当前帧（速度很慢）
	# cv2.setTrackbarPos('time', 'video', int(current_pos))
	img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
	# 若和bar同一个窗口显示，速度很慢
	# cv2.imshow("video", img)
	cv2.imshow("result",img)

	# press keyboard 'q' to exit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
	time2 = time.time()
	print("time2-time1 =", time2 - time1)