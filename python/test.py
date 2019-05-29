import cv2
import time

video = '/Volumes/MachsionHD/drone_实验视频/fixed-wing.mp4'
cv2.namedWindow('video')
cap = cv2.VideoCapture(video)

while 1:
    time1 = time.time()
    ret,frame = cap.read()
    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time2 = time.time()
    print("time2-time1 =", time2 - time1)