import numpy as np
import cv2, time
import tools
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


###################################### 是否显示检测结果 ##################################################
is_show = True
###################### threshold_gain ##################################################
diff_threshhold = 25

video = '/Volumes/MachsionHD/drone_实验视频/实验原始视频（部分转换格式）/20190403紫金港/20190403紫金港300m-实验3/20190403_20190403150127_20190403150647_150127-ptz白.mp4'
# video = '/Volumes/MachsionHD/drone_实验视频/实验原始视频（部分转换格式）/20190403紫金港/20190403紫金港450m-实验2/20190403_20190403144034_20190403145700_144034-全景.mp4'
cap = cv2.VideoCapture(video)
startframe = 10000
frame_num = startframe
endframe = 18000#int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


if is_show:
    cv2.namedWindow('Original Frame')
    cv2.namedWindow('Normalized Interframe Difference Map')
    cv2.namedWindow('Saliency Map')


##############################目标检测开始#################################################


while(cap.isOpened()):

    print("frame_num:%d " % frame_num)

    ############################## 0 read frames ##############################################
    # 若是第一帧
    if frame_num == startframe:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, pre_BGR_frame = cap.read()  # read image
        # 降采样、转化灰度图、高斯滤波
        # pre_BGR_frame = getDownsamplingImg(pre_BGR_frame)
        pre_frame = cv2.cvtColor(pre_BGR_frame, cv2.COLOR_BGR2GRAY)  # convert RGB -> gray
        # pre_frame = pre_frame[0: 1200, :]  # delete the ground backgroud


    # 若非第一帧
    if frame_num > startframe and frame_num <= endframe:
        ret, cur_BGR_frame = cap.read()
        cur_frame = cv2.cvtColor(cur_BGR_frame, cv2.COLOR_BGR2GRAY)  # convert RGB -> gray

        # 帧间差分
        diffMap, cnts_diff = tools.frame_diff(cur_frame, pre_frame, diff_threshhold)
        diff_BGR_frame = tools.drawRectangle(cnts_diff, cur_BGR_frame, (0, 0, 255))
        cv2.imshow('Normalized Interframe Difference Map', tools.getDownsamplingImg(diffMap,0.3))
        cv2.imshow("Original Frame", tools.getDownsamplingImg(diff_BGR_frame,0.3))

        # 谱残差
        saliencyMap = tools.SR(pre_frame)
        cv2.imshow('Saliency Map', tools.getDownsamplingImg(saliencyMap, 0.3))

        # 更新背景
        pre_frame = cur_frame

    frame_num += 1

    # press keyboard 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()