import darknet as dn
import time
import random
import numpy as np
import cv2
import gc

dn.set_gpu(0)
net = dn.load_net(str.encode("drone/yolov3-drone1-tiny.cfg"),
                  str.encode("drone/yolov3-drone1-tiny_10200.weights"), 0)
meta = dn.load_meta(str.encode("drone/drone1.data"))
# r = dn.detect(net,meta,str.encode("drone/drone.png"))
# print(r)


def detect2(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    # num = dn.c_int(0)
    # pnum = dn.pointer(num)
    # dn.predict_image(net, image)
    # dets = dn.get_network_boxes(net, image.w, image.h, thresh, hier_thresh, None, 0, pnum)
    # num = pnum[0]
    # if (nms): dn.do_nms_obj(dets, num, meta.classes, nms);

    # res = []
    # for j in range(num):
    #     for i in range(meta.classes):
    #         if dets[j].prob[i] > 0:
    #             b = dets[j].bbox
    #             res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    # res = sorted(res, key=lambda x: -x[1])
    # dn.free_image(image)
    # dn.free_detections(dets, num)
    # return res
    num = dn.c_int(0)
    pnum = dn.pointer(num)
    dn.predict_image(net, image)
    dets = dn.get_network_boxes(net, image.w, image.h, thresh, 
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: dn.do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], 
                           (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): 
        dn.free_image(image) #这步什么情况下执行？多次无法执行的原因是未对这句做判断
        print("free image") 
    # dn.free_image(image)
    dn.free_detections(dets, num)
    return res

def pipeline(img):
    # image data transform
    # img - cv image
    # im - yolo image
    im, image = dn.array_to_image(img)
    dn.rgbgr_image(im)

    tic = time.time()
    result = detect2(net, meta, im)
    toc = time.time()
    print(toc - tic, result)

    img_final = dn.draw_boxes(img, result)
    return img_final

im = cv2.imread("drone/drone.png")
cv2.imshow("Image", im)
cv2.waitKey(1)

sp = im.shape #height/width/
overlap = [20,30]
height = int(sp[0]/2+overlap[0])
width = int(sp[1]/2+overlap[1])

cropImg1 = im[0:height,0:width]
cropImg2 = im[0:height,-width:-1]
cropImg3 = im[-height:-1,0:width]
cropImg4 = im[-height:-1,-width:-1]


for cropImg in [cropImg1,cropImg2,cropImg3,cropImg4]:
    cv2.imshow("YOLO", pipeline(cropImg))
    cv2.waitKey(1)
# cv2.imshow("YOLO", pipeline(cropImg2))

# exit code -6/arr出错
# for cropImg in [cropImg1,cropImg2,cropImg3,cropImg4]:# 
#     cropIm,arr = dn.array_to_image(cropImg)
#     dn.rgbgr_image(cropIm)
#     r = detect2(net, meta, cropIm)
#     # cv2.imshow("YOLO",cropImg)
#     # cv2.waitKey(0)
#     cv2.imshow("YOLO", dn.draw_boxes(cropImg, r))
#     print(r)

# ----------------可运行----------------
# cropIm,arr1 = dn.array_to_image(cropImg1)
# dn.rgbgr_image(cropIm)
# r = detect2(net, meta, cropIm)
# cv2.imshow("YOLO", dn.draw_boxes(cropImg1, r))
# print(r)


# cropIm,arr2 = dn.array_to_image(cropImg2)
# dn.rgbgr_image(cropIm)
# r = detect2(net, meta, cropIm)
# cv2.imshow("YOLO", dn.draw_boxes(cropImg2, r))
# print(r)
cv2.waitKey(0)
cv2.destroyAllWindows()

