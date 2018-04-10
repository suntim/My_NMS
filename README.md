# My_NMS


# -*- coding: UTF-8 -*-
#!/usr/bin/env python
from lxml import etree
import os
import re
import cv2
import numpy as np

def head_info(image_name,save_dir,im_w,im_h,im_depth=1):
    data = etree.Element("annotation")
    data.set('verified', 'no')
    # 1 folder
    interface_folder = etree.SubElement(data, 'folder')
    interface_folder.text = 'testXT'
    # 2 filename
    filename_txt = image_name
    filename = etree.SubElement(data, 'filename')
    filename.text = filename_txt
    # 3 path
    pathNode = etree.SubElement(data, 'path')
    pathNode.text = os.path.join(save_dir , filename_txt + '.jpg')
    # 4 source
    source = etree.SubElement(data, 'source')
    database = etree.SubElement(source, 'database')
    database.text = 'Unknown'
    # 5 img size
    imgsize = etree.SubElement(data, 'size')
    img_width = etree.SubElement(imgsize, 'width')
    img_width.text = str(im_w)
    img_height = etree.SubElement(imgsize, 'height')
    img_height.text = str(im_h)
    img_depth = etree.SubElement(imgsize, 'depth')
    img_depth.text = str(im_depth)
    # 6 segmented
    segmented = etree.SubElement(data, 'segmented')
    segmented.text = '0'
    return data


def object_info(data, objName, x1, y1, x2, y2,x3,y3,x4,y4):
    object = etree.SubElement(data, 'object')
    object_name = etree.SubElement(object, 'name')
    object_name.text = objName
    pose = etree.SubElement(object, 'pose')
    pose.text = 'HikPolygonRoiParameter'
    truncated = etree.SubElement(object, 'truncated')
    truncated.text = '0'
    difficult = etree.SubElement(object, 'difficult')
    difficult.text = '0'
    bndbox = etree.SubElement(object, 'bndbox')
    X = etree.SubElement(bndbox, 'X')
    X.text =x1
    Y = etree.SubElement(bndbox, 'Y')
    Y.text = y1
    X = etree.SubElement(bndbox, 'X')
    X.text = x2
    Y = etree.SubElement(bndbox, 'Y')
    Y.text = y2
    X = etree.SubElement(bndbox, 'X')
    X.text = x3
    Y = etree.SubElement(bndbox, 'Y')
    Y.text = y3
    X = etree.SubElement(bndbox, 'X')
    X.text = x4
    Y = etree.SubElement(bndbox, 'Y')
    Y.text = y4





if __name__ == '__main__':
    class_name = "1"
    img_dir = r"D:\label_1"
    detect_txt_dir = r"D:\label_1\results_20180404_142948"
    save_dir = r"D:\label_1\Xml"
    for fileName in os.listdir(img_dir):
        if re.match(".*[.]jpg$",fileName):
            detect_txt_path = os.path.join(detect_txt_dir,"res_"+fileName.split('.')[0]+'.txt')
            assert os.path.exists(detect_txt_path),"detect_txt_path = {} Not Existed!!!".format(detect_txt_path)
            print detect_txt_path
            # Load the demo image
            im_file = os.path.join(img_dir, fileName)
            im = cv2.imread(im_file)
            im_h, im_w, im_depth = im.shape[:3]
            # print im_h, im_w, im_depth
            data = head_info(fileName.split('.')[0],save_dir,im_w,im_h,im_depth);
            # 7 object
            with open(detect_txt_path,'r') as f:
                for line in f.readlines():
                    ss = line.strip().split(",")
                    [x1, y1, x2, y2, x3, y3, x4, y4] = ss
                    print x1, y1, x2, y2,x3,y3,x4,y4
                    object_info(data, objName=class_name, x1=x1, y1=y1, x2=x2, y2=y2,x3=x3,y3=y3,x4=x4,y4=y4)

            # 8 write xml
            dataxml = etree.tostring(data, pretty_print=True, encoding="UTF-8", method="xml", xml_declaration=True,
                                     standalone=None)
            with open(os.path.join(save_dir, (fileName.split('.')[0] + '.xml')), 'w') as fp:
                fp.write(dataxml)
