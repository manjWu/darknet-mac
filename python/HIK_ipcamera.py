# // 说明：
# // username：用户名，例如admin
# // passwd：密码，例如12345
# // ip：设备的ip地址，例如192.0.0.64
# // port：端口号默认554，若为默认可以不写
# // codec：有h264、MPEG-4、mpeg4这几种
# // channel：通道号，起始为1
# // subtype：码流类型，主码流为main，子码流为sub

# rtsp://[username]:[passwd]@[ip]:[port]/[codec]/[channel]/[subtype]/av_stream

import cv2

cam = cv2.VideoCapture("rtsp://admin:nesc518518@10.15.90.65:554/h264/ch1/sub/av_stream")
# cam = cv2.VideoCapture("rtsp://admin:nesc518518@10.15.90.64:554/Streaming/Channels/101")
while True:
    ret, frame = cam.read()
    if ret == True: #termino los frames?
        cv2.imshow("test", frame)
        key = cv2.waitKey(100) & 0xff
        if key == 27: # ESC
            break

cam.release()
cv2.destroyAllWindows()