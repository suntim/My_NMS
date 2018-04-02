# My_NMS

Read Xml and detect the object
# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import numpy as np
import cv2
import os
import math
import re

def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

def BoxRectangMethod(Area_Rect,proimage,Method):
    """
    :param Area_Rect:
    :param proimage:
    :param Method:
    :return: bbx =[]#x1,x2,y1,y2,angle
    """
    bbx =[]#x1,x2,y1,y2
    if Method == 0:
        x, y, w, h = cv2.boundingRect(Area_Rect)
        rect = cv2.minAreaRect(Area_Rect)
        angle = rect[-1]
        # print "x = {}, y= {}, w = {}, h = {}".format(x, y, w, h)
        bbx.append(x),bbx.append(y),bbx.append(x+w),bbx.append(y+h),bbx.append(angle)
        # cv2.rectangle(proimage, (x, y), (x + w, y + h), (0, 0, 255), 3)
        # cv2.imshow("Image", proimage)
        # cv2.waitKey()
    elif Method == 1:
        rect = cv2.minAreaRect(Area_Rect)
        angle = rect[-1]
        box = np.int0(cv2.boxPoints(rect))  # cv2.cv.BoxPoints(rect)
        Xs = [i[0] if i[0] > 0 else 0 for i in box]
        Ys = [i[1] if i[1] > 0 else 0 for i in box]
        print " Xs = ", Xs
        # draw a bounding box arounded the detected barcode and display the image
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        bbx.append(x1), bbx.append(y1), bbx.append(x2), bbx.append(y2),bbx.append(angle)
        # cv2.rectangle(proimage, (x1,y1), (x2,y2),(0, 255, 0), 3)
        # cv2.imshow("Image", proimage)
        # cv2.waitKey()
    elif Method == 2:
        rect = cv2.minAreaRect(Area_Rect)
        box = np.int0(cv2.boxPoints(rect))  # cv2.cv.BoxPoints(rect)
        Xs = []
        Ys = []
        for i in box:
            if i[0] > 0 and i[0] < proimage.shape[1]:
                Xs.append(i[0])
            elif i[0] < 0:
                Xs.append(0)
            elif i[0]>proimage.shape[1]:
                Xs.append(int(proimage.shape[1])-2)
                print "proimage.shape[1] = ",proimage.shape[1]

            if i[1] > 0 and i[1] < proimage.shape[0]:
                Ys.append(i[1])
            elif i[1] < 0:
                Ys.append(0)
            elif i[1] > proimage.shape[0]:
                Ys.append(int(proimage.shape[0])-2)
                print "proimage.shape[0] = ", proimage.shape[0]

        # print "rect[-1] = ",rect[-1]
        print "Xs = ",Xs
        print "Ys = ",Ys
        angle = rect[-1]
        bbx.append(Xs),bbx.append(Ys),bbx.append(angle)
        cv2.drawContours(proimage, [box], -1, (255, 0, 0), 3)
        cv2.imshow("Image", proimage)
        cv2.waitKey()
    if Method == "01":
        x, y, w, h = cv2.boundingRect(Area_Rect)
        rect = cv2.minAreaRect(Area_Rect)
        # print "x = {}, y= {}, w = {}, h = {}".format(x, y, w, h)
        cv2.rectangle(proimage, (x, y), (x + w, y + h), (0, 0, 255), 3)

        box = np.int0(cv2.boxPoints(rect))  # cv2.cv.BoxPoints(rect)
        Xs = [i[0] if i[0] > 0 else 0 for i in box]
        Ys = [i[1] if i[1] > 0 else 0 for i in box]
        # draw a bounding box arounded the detected barcode and display the image
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)

        cv2.rectangle(proimage, (x1,y1), (x2,y2),(0, 255, 0), 3)
        cv2.imshow("Image", proimage)
        cv2.waitKey()
    return bbx

def ErodeAndDilate(label_thresh):
    """
    进行膨胀和腐蚀处理
    :param label_thresh: 二值图
    :return:处理的二值图结果
    """
    """腐蚀"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(label_thresh, cv2.MORPH_CLOSE, kernel)
    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=10)
    closed = cv2.dilate(closed, None, iterations=1)
    # cv2.imshow("closed",closed)
    return closed

def SegRoIFunc(img_path,detect_path,save_dir,Method):
    """
    提取轮廓
    :param img_path:
    :param detect_path:
    :param save_dir:
    :param Method:
    :return:
    """
    """src为原图"""
    src = cv2.imread(img_path)
    proimage = src.copy()         #复制原图
    ROI =  cv2.imread(detect_path)    #感兴趣区域ROI

    """提取轮廓"""
    # label = cv2.bitwise_not(ROI)
    label = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("label",label)
    thresh, label_thresh = cv2.threshold(label, 10, 255, cv2.THRESH_BINARY)
    # cv2.imshow("label_thresh",label_thresh)

    closed = ErodeAndDilate(label_thresh)

    """寻找最小面积的矩形"""
    (_, cnts, _) = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print cnts
    c = sorted(cnts, key=cv2.contourArea, reverse=True)

    for i in range(np.array(c).shape[0]):
        save_path = os.path.join(save_dir,"Method={}".format(Method))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # compute the rotated bounding box of the largest contour
        imgName = img_path.split("\\")[-1].split('.')[0]
        # print imgName
        if Method != 2 and Method != "01":
            bbx = BoxRectangMethod(c[i], proimage, Method)
            cropImg = src[bbx[1]:bbx[3], bbx[0]:bbx[2]]
            cv2.imwrite(os.path.join(save_path,"{}_{}.jpg".format(imgName,i)),cropImg)
            cv2.imshow("cropImg",cropImg)
            cv2.waitKey()
            angle = bbx[4]
            cropRotateImg = rotate_about_center(cropImg, angle, scale=1.)
            cv2.imwrite(os.path.join(save_path, "cropRotateImg_{}_{}.jpg".format(imgName, i)), cropRotateImg)
            cv2.imshow("cropRotateImg", cropRotateImg)
            cv2.waitKey()
        elif Method == 2 :
            bbx = BoxRectangMethod(c[i], proimage, Method)
            origin_selected_conors = []
            rigin_selected_lu = (bbx[0][2], bbx[1][2])  # left up
            print "rigin_selected_lu = ", rigin_selected_lu
            rigin_selected_ru = (bbx[0][3], bbx[1][3])  # right up
            print "rigin_selected_ru = ",rigin_selected_ru
            rigin_selected_ld = (bbx[0][1], bbx[1][1])  # left down
            print "rigin_selected_ld = ",rigin_selected_ld
            rigin_selected_rd = (bbx[0][0], bbx[1][0])  # right down
            print "rigin_selected_rd = ",rigin_selected_rd

            # 添加到 origin_selected_conors
            # rigin_selected_lu = (724, 105)
            # rigin_selected_ru = (764, 210)
            # rigin_selected_ld = (514, 185)
            # rigin_selected_rd = (554, 290)
            origin_selected_conors.append(rigin_selected_lu)
            origin_selected_conors.append(rigin_selected_ru)
            origin_selected_conors.append(rigin_selected_rd)
            origin_selected_conors.append(rigin_selected_ld)

            # 变换过后图像展示在 一个 宽为 show_width 长为 show_height的长方形窗口
            # print "rigin_selected_lu[0] = ",rigin_selected_lu[0]
            show_width1 = int(((rigin_selected_lu[0]-rigin_selected_ru[0])**2+(rigin_selected_lu[1]-rigin_selected_ru[1])**2)**0.5)
            show_width2 = int(((rigin_selected_ld[0]-rigin_selected_rd[0])**2+(rigin_selected_ld[1]-rigin_selected_rd[1])**2)**0.5)
            show_width = max(show_width1,show_width2)
            print "show_width = ",show_width

            # print "rigin_selected_rd[0] = ", rigin_selected_rd[0]
            show_height1 = int(((rigin_selected_ru[0] - rigin_selected_rd[0]) ** 2 + (rigin_selected_ru[1] - rigin_selected_rd[1]) ** 2) ** 0.5)
            show_height2 = int(((rigin_selected_ru[0] - rigin_selected_rd[0]) ** 2 + (rigin_selected_ru[1] - rigin_selected_rd[1]) ** 2) ** 0.5)
            show_height = max(show_height1,show_height2)
            print "show_height = ", show_height
            show_window_conors = []
            show_window_lu = (0, 0)
            show_window_ru = (show_width - 1, 0)
            show_window_ld = (0, show_height - 1)
            show_window_rd = (show_width - 1, show_height - 1)

            # 添加到 show_window_conors中
            show_window_conors.append(show_window_lu)
            show_window_conors.append(show_window_ru)
            show_window_conors.append(show_window_rd)
            show_window_conors.append(show_window_ld)

            # 获得transform 函数
            transform = cv2.getPerspectiveTransform(np.array(show_window_conors, dtype=np.float32),
                                                    np.array(origin_selected_conors, dtype=np.float32))

            transfered_pos = np.zeros([show_width, show_height, 2])
            for x in range(show_width):
                for y in range(show_height):



# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import numpy as np
import cv2
import sys
show_width = 322
show_height = 200

global opened_pic_file
global star_points
# global select_point_num
star_points = []


def  onMouse(event, x, y, flag, param):
    select_point_num = 0
    if event == 4 and select_point_num <4:
        print x, y, select_point_num,

        # 已选择的点加 1
        select_point_num = select_point_num + 1

        # 将选择好的点添加到相应的数组当中
        point = (x,y)
        cv2.circle(img, point, 2, (0, 255, 0), 2)#修改最后一个参数

        # 划线
        if len(star_points) >= 1:
            # 取出最后一个点
            last_point = star_points[len(star_points)-1]
            # 划线
            cv2.line(img, point, last_point, (155, 155, 155), 2)

        if len(star_points) == 3:
            # 取出开始的一个点
            last_point = star_points[0]
            # 划线
            cv2.line(img, point, last_point, (155, 155, 155), 2)

        # 更新图片
        cv2.imshow(window, img)
        star_points.append(point)
        if len(star_points) == 4:
            rectify_that_part_of_photo()

def  rectify_that_part_of_photo():
    # 打开一份备份img
    img_copy = cv2.imread(opened_pic_file)
    cv2.namedWindow("result_img", 0)
    print "star_points = ",star_points

    origin_selected_conors = []
    rigin_selected_lu = (star_points[0][0],star_points[0][1])#left up
    print "rigin_selected_lu = ",rigin_selected_lu
    rigin_selected_ru = (star_points[1][0],star_points[1][1])#right up
    rigin_selected_ld = (star_points[3][0],star_points[3][1])#left down
    rigin_selected_rd = (star_points[2][0],star_points[2][1])#right down

    # 添加到 origin_selected_conors
    origin_selected_conors.append(rigin_selected_lu)
    origin_selected_conors.append(rigin_selected_ru)
    origin_selected_conors.append(rigin_selected_rd)
    origin_selected_conors.append(rigin_selected_ld)

    # 变换过后图像展示在 一个 宽为 show_width 长为 show_height的长方形窗口
    show_window_conors = []
    show_window_lu = (0, 0)
    show_window_ru = (show_width-1, 0)
    show_window_ld = (0, show_height-1)
    show_window_rd = (show_width-1, show_height-1)

    # 添加到 show_window_conors中
    show_window_conors.append(show_window_lu)
    show_window_conors.append(show_window_ru)
    show_window_conors.append(show_window_rd)
    show_window_conors.append(show_window_ld)

    # 获得transform 函数
    transform = cv2.getPerspectiveTransform(np.array(show_window_conors, dtype=np.float32), np.array(origin_selected_conors, dtype=np.float32))

    transfered_pos = np.zeros([show_width, show_height, 2])
    for x in range(show_width):
        for y in range(show_height):
            temp_pos = np.dot(transform, np.array([x, y, 1]).T)
            transed_x = temp_pos[0]/temp_pos[2]
            transed_y = temp_pos[1]/temp_pos[2]
            transfered_pos[x][y] = (int(transed_x), int(transed_y))

    # 生成 一个空的彩色图像
    result_img = np.zeros((show_height, show_width, 3), np.uint8)
    print result_img.shape

    for x in range(show_width):
        for y in range(show_height):
            # print "transfered_pos[x][y][1] = ",transfered_pos[x][y][1]
            result_img[y][x] = img_copy[int(transfered_pos[x][y][1])][int(transfered_pos[x][y][0])]

    cv2.imshow("result_img", result_img)


if __name__ == '__main__':
    # 获取用户的输入
    # opened_pic_file 输入的图片地址和文件名
    # if len(sys.argv) != 2:
    #     print "please input the filename!!!"
    #     exit(0)
    # else:
    #     opened_pic_file = sys.argv[1]

    opened_pic_file = r'D:\Bill\TrainSegBill\test\orig\00006.jpg'
    window = "window"
    img = cv2.imread(opened_pic_file)
    img2 = []

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    cv2.imshow(window, img)

    # 2. 给用户注册鼠标点击事件
    cv2.setMouseCallback(window, onMouse, None)

    # 监听用户的输入，如果用户按了esc建，那么就将window给销毁
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyWindow(window)
