import numpy as np
import cv2, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###################################### flag flag flag###################################################
startframe = 0
endframe = 1478
###################################### 是否显示检测结果 ##################################################
is_show = True
###################################### 是否显示检测结果 ##################################################
is_heat_map = False
heatmap_save_file_path = '/Users/shallwego2/MyGit/ST_Fusion_object_detection/20190411_ST3_heatmap/'
###################################### 用两帧差分还是三帧差分 #############################################
is_3 = False
###################################### 是否多目标跟踪 ####################################################
is_tracking = False
###################################### 是否计算recall与准确率 ############################################
Test = True
###################################### threshold_gain ##################################################
threshold_gain = 1000


################### use global_threhold or adaptive local threshold
global_threshold = True

################## if 12 < area < 400 and w > 6 and h > 3 ##############################################
min_area = 12
max_area = 400
min_width = 6
min_height = 3
max_width = 100
max_height = 100

###################### use which type of morphological method #############################################
morphological_method = 'closing'

###################################### kernel size #########################################################
kernel = np.ones((5, 5), np.uint8)

################### flag for RGB or gray ##################################################################
is_gray = True

################### read image sequences
image_sequence_file = '/Users/shallwego2/MyGit/ST_Fusion_object_detection/grassland/uav_image_list.txt'  # this txt file contain all the image file names



################### the dataset path
Path2_BGRTimage_dataset = '/Users/shallwego2/MyGit/ST_Fusion_object_detection/grassland/T_images_BGR/'
Path2_grayTimage_dataset = '/Users/shallwego2/MyGit/ST_Fusion_object_detection/grassland/T_images_gray/'

# value setting for filter size and kernel size
filter_size = 5
kernel_size = 5

IsLog = True
is_th_scan = False
##################################### read labels ######################################################
label_list_path = '/Users/shallwego2/MyGit/ST_Fusion_object_detection/grassland/labels_txt/labels_txt_list.txt'
with open(label_list_path) as f:
    label_file_list = f.readlines()



############################### 归一化检测图 ##########################################################
def Normalization_plus(M):
    '''
    :param M:
    :return: a max normalization matrix
    '''
    M_shape = M.shape

    if M.ndim == 2:
        max_value = M.max()
        M = M / max_value
    elif M_shape[2] == 3:
        max_R = np.max(M[:, :, 0])
        max_G = np.max(M[:, :, 1])
        max_B = np.max(M[:, :, 2])
        M[:, :, 0] = M[:, :, 0] / max_R
        M[:, :, 1] = M[:, :, 1] / max_G
        M[:, :, 2] = M[:, :, 2] / max_B
    else:
        print("The image channel is wrong!")

    return M






start = time.time()


M1_every_frame_recall = []
M1_every_frame_precision = []

M1_sequence_recall = 0
M1_sequence_precision = 0



# 真实目标的总数目
total_num_of_real_uav = 0

M1_total_num_of_detection = 0


M1_TP = 0


M1_boxes = []


##################################### Method name ######################################################
M1 = 1


if is_show:
    cv2.namedWindow('Original Frame', cv2.WINDOW_NORMAL)


    cv2.namedWindow('Normalized Interframe difference map', cv2.WINDOW_NORMAL)

th = 100

# read the image sequence file names
with open(image_sequence_file) as f:
    img_name_list = f.readlines()
    ###############################################       目标检测开始       #################################################

for frame_num, image_name in enumerate(img_name_list):
    image_name = image_name.rstrip()  # strip the enter(\n)

    print("frame_num:%d " % frame_num)


    ############################## 0 read frames ##############################################
    if frame_num == startframe:
        pre_BGR_frame = cv2.imread(image_name)  # read image
        if is_gray:
            pre_frame = cv2.cvtColor(pre_BGR_frame, cv2.COLOR_BGR2GRAY)  # convert RGB -> gray
            pre_frame = pre_frame[0: 1200, :]  # delete the ground backgroud
        else:
            pre_frame = pre_BGR_frame[0:1200, :]

    if frame_num > startframe and frame_num <= endframe:
        cur_BGR_frame = cv2.imread(image_name)
        cur_frame = cv2.cvtColor(cur_BGR_frame, cv2.COLOR_BGR2GRAY)  # convert RGB -> gray
        cur_frame = cur_frame[0: 1200, :]  # delete the ground backgroud


        diff_frame_uint8 = cv2.absdiff(cur_frame, pre_frame)
        # interframe difference and convert its type from np.unit8 to np.float32
        diff_frame = cv2.absdiff(cur_frame, pre_frame).astype(np.float32)
        # max_normalize the diff_gray_frame
        N_float32_diff_frame = Normalization_plus(diff_frame) * 255
        N_diff_frame = np.array(N_float32_diff_frame, dtype=np.uint8)
        # 更新背景
        pre_frame = cur_frame

        avg_1 = np.mean(N_diff_frame)


        threshold1 = threshold_gain * avg_1 if threshold_gain * avg_1 < th else th

        bin_diff_frame = cv2.threshold(N_diff_frame, threshold1, 255, cv2.THRESH_BINARY)[1]
        bin_diff_frame = cv2.dilate(bin_diff_frame, None, iterations=2)

        (_, cnts_M1, _) = cv2.findContours(bin_diff_frame.copy(), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_TC89_KCOS)

        for contour in cnts_M1:
            (x, y, w, h) = cv2.boundingRect(contour)
            area = w * h
            if min_area < area < max_area and max_width > w > min_width and max_height > h > min_height:
                M1_boxes.append(cv2.boundingRect(contour))
            for i in range(len(M1_boxes)):
                x = M1_boxes[i][0]
                y = M1_boxes[i][1]
                w = M1_boxes[i][2]
                h = M1_boxes[i][3]
                cv2.rectangle(cur_BGR_frame, (x, y), (x + w, y + h), (0, 0, 0),1)  # Drawing rectangle over objects
                cv2.rectangle(cur_BGR_frame, (x, y), (x + w, y + h), (0, 255, 0),1)  # Drawing rectangle over objects
            cv2.imshow("Normalized Interframe difference map", bin_diff_frame)
            cv2.imshow("Original Frame", cur_BGR_frame)
            cv2.waitKey(10)
