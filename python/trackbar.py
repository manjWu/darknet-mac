
'''
使用r/s实现视频+滑块的单步模式播放以及持续播放（滑块随动）
'''



import cv2
import time

video = '/Volumes/MachsionHD/drone_实验视频/fixed-wing.mp4'
cv2.namedWindow('video')
cap = cv2.VideoCapture(video)
# 当前滑块的位置
current_pos = 0
run_flag = 1
dontset_flag = 0
# 获得视频总帧数
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame:",frames)

# 在轨迹条位置改变的时候来调用回调函数
# 滑块位置主动: 移动滑块，读取滑块位置，再读取视频
def onTrackbarslide(pos):
	# 设置视频帧的读取位置
	print("onTrackbarslide...")
	cap.set(cv2.CAP_PROP_POS_FRAMES,pos)
	global run_flag
	global dontset_flag
	if dontset_flag==0:
		run_flag = 1
		dontset_flag = 0



# 创建滑块
cv2.createTrackbar('time', 'video', 0, frames, onTrackbarslide)



while 1:
	if run_flag!=0:
		#滑块位置随动: 读视频帧数，再移动滑块
		ret, frame = cap.read()
		# 获取当前视频帧数
		current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
		print("current_pos:",current_pos)
		dontset_flag = 1 # 使得下一个callback函数不会将系统置于单步模式
		# 将滑块移动到当前帧
		cv2.setTrackbarPos('time', 'video', int(current_pos))
		img = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
		cv2.imshow("video",img)

		run_flag -= 1
		print("run_flag =",run_flag)

	if cv2.waitKey(1) & 0xFF == ord('s'):
		run_flag = 1
		print("Single step, run =", run_flag)
	if cv2.waitKey(1) & 0xFF == ord('d'):
		run_flag = -1
		print("Run mode, run =", run_flag)
	# press keyboard 'q' to exit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
