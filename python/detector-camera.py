import time
import random
import numpy as np
import cv2
import darknet as dn


# import colorsys
# from PIL import Image, ImageDraw, ImageFont


# prepare YOLO
net = dn.load_net(str.encode("../cfg/yolov3-tiny.cfg"),
                  str.encode("../drone/yolov3-tiny.weights"), 0)
meta = dn.load_meta(str.encode("../cfg/coco.data"))
# net = dn.load_net(str.encode("drone/yolov3-drone1-tiny.cfg"),
#                   str.encode("drone/yolov3-drone1-tiny_10200.weights"), 0)
# meta = dn.load_meta(str.encode("drone/drone1.data"))

# def detect1(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
#     boxes = dn.make_network_boxes(net)
#     probs = dn.make_network_boxes(net)
#     num = dn.num_boxes(net)
#     dn.network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
#     res = []
#     for j in range(num):
#         for i in range(meta.classes):
#             if probs[j][i] > 0:
#                 res.append(
#                     (meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
#     res = sorted(res, key=lambda x: -x[1])
#     dn.free_ptrs(dn.cast(probs, dn.POINTER(dn.c_void_p)), num)
#     return res


def detect2(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    """if isinstance(image, bytes):  
        # image is a filename 
        # i.e. image = b'/darknet/data/dog.jpg'
        im = load_image(image, 0, 0)
    else:  
        # image is an nparray
        # i.e. image = cv2.imread('/darknet/data/dog.jpg')
        im, image = array_to_image(image)
        rgbgr_image(im)
    """
    # im, image = array_to_image(image)
    # rgbgr_image(im)
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
        dn.free_image(image)
        print("free_image")#????
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


count_frame, process_every_n_frame = 0, 10
# get camera device
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("rtsp://admin:nesc518518@10.15.90.64:554/h264/ch1/sub/av_stream")

while(True):
    # get a frame
    ret, frame = cap.read()
    count_frame += 1

    # show a frame
    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # resize image half
    cv2.imshow("Video", img)

    # if running slow on your computer, try process_every_n_frame = 10
    if count_frame % process_every_n_frame == 0:
        print("yolo detecting...")
        cv2.imshow("YOLO", pipeline(img))

    # press keyboard 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
