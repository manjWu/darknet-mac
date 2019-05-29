# -*- coding: utf-8 -*-

"""
Created on Thu Jan 10 16:21:08 2019
背景差分法：pre_frame - cur_frame
measureType: 0:直接差值度量；1:基于特征相似度度量
函数：frame_diff
输入：cur_BGR_frame（当前帧原始图像）,pre_frame（灰度图）, diff_threshhold
输出：cur_frame（当前帧灰度图）
@author: wumanjia
"""

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



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

def getDownsamplingImg(img, para=0.5):
    # 降采样 减少运算量
    img = cv2.resize(img,None, fx=para,fy=para,interpolation = cv2.INTER_AREA)
    return img

def drawRectangle(cnts,img, rect_color):
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        print("检测到运动区域大小为：",w * h)
        if w * h > 10 and w * h < 200:
            # 连通域要足够大才标box
            cv2.rectangle(img, (x, y), (x + w, y + h),rect_color, 1)
            # 图片，添加的文字，左上角坐标(整数)，字体，字体大小，颜色，字体粗细
            cv2.putText(img, 'Object', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.3, rect_color, 1)
        else:
            pass
            # cv2.rectangle(img, (x, y), (x + w, y + h),(255,0,0), 1)
            # cv2.putText(img, 'Pseudo', (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 0, 0), 1)
    return img

def im2double(im):
    # min_val = np.min(im.ravel())
    # max_val = np.max(im.ravel())
    # out = (im.astype('double') - min_val) / (max_val - min_val)
    out = im.astype('double') /255
    return out


############################### SR ##########################################################
def SR(inImg):
    # im2double 将图像转化为双精度值
    # 将彩色图像转为灰度图像（即亮度）后进行二维离散傅立叶变换
    # inImg = cv2.cvtColor(inImg, cv2.COLOR_BGR2GRAY)
    inImg = im2double(inImg)
    # inImg = cv2.normalize(inImg.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)  # Convert to normalized floating point
    # Spectral Residual
    myFFT = np.fft.fft2(inImg)
    myLogAmplitude = np.log(abs(myFFT))
    myPhase = np.angle(myFFT)
    # 创建局部平均滤波算子[3，3]为默认尺寸
    mySmooth = cv2.blur(myLogAmplitude, (3, 3), borderType=1)
    # 幅度值的Log谱 - 局部平均滤波器进行平滑处理后的Log谱得到普残差
    mySpectralResidual = myLogAmplitude - mySmooth
    # 将相位谱和谱残差进行二维傅立叶反变换得到显著图
    saliencyMap = np.power(abs(np.fft.ifft2(np.exp(mySpectralResidual + 1j * myPhase))), 2)
    # 对处理后的显著图进行滤波（高斯低通滤波尺寸为[3，3]，Sigma为滤波器的标准差）
    saliencyMap = cv2.blur(saliencyMap, (3, 3), borderType=0)  # sigmaX表示X方向方差。注意，核大小（N, N）必须是奇数，X方向方差主要控制权重。
    # 然后归一化
    saliencyMap = cv2.normalize(saliencyMap, None, 0.0, 1.0, cv2.NORM_MINMAX)

    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    saliencyMap = cv2.dilate(saliencyMap, es, iterations=2)  # 膨胀

    return saliencyMap

############################### frame_diff ##########################################################
def frame_diff(cur_frame,pre_frame, diff_threshhold):# 输入灰度图
    # measureType = int(input('plz input a measureType: 0:直接差值度量；1:基于余弦相似度度量（余弦相似度度量还没写，先用0）'));
    measureType = 0
    # 定义形态学运算的核元素
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    # 10x10
    # 定义椭圆（MORPH_ELLIPSE）和十字形结构（MORPH_CROSS）元素

    # 降采样、转化灰度图、高斯滤波
    # cur_BGR_frame = getDownsamplingImg(cur_BGR_frame)
    # cur_frame = cv2.cvtColor(cur_BGR_frame, cv2.COLOR_BGR2GRAY)  # convert RGB -> gray
    # cur_frame = cur_frame[0: 1200, :]  # delete the ground backgroud
    # cur_frame = cv2.GaussianBlur(cur_frame,(21,21),0)

    #进行图像差分、二值化、形态学运算
    diff = cv2.absdiff(pre_frame, cur_frame)
    diff = cv2.threshold(diff, diff_threshhold, 255, cv2.THRESH_BINARY)[1]
    # 愿图像，进行分类的阈值，高于/低于阈值赋予新值，方法选择参数，黑白二值，_inv黑白二值反转
    # 返回值1阈值值，返回值2阈值化后的图像
    diff = cv2.dilate(diff, es, iterations=2 )#膨胀

    #获取轮廓线
    # image, cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (_, cnts, _) = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    # 参数意义：输入图像。轮廓检索模式，轮廓近似方法


    return diff, cnts

