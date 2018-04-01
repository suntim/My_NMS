# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import os
import re
import cv2
from pylab import matplotlib
zhfont = matplotlib.font_manager.FontProperties(fname = "/usr/share/fonts/truetype/arphic/ukai.ttc")

def py_cpu_nms(dets, ovr_thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    print "order = ",order
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # print "0vr = ",ovr

        inds = np.where(ovr <= ovr_thresh)[0]
        # print "inds(ovr <= ovr_thresh[{}]) = {}".format(ovr_thresh,inds)
        order = order[inds + 1]
    return keep

def parse_rec(filename):
    """
    一个xml所有的objects信息
    :param filename:
    :return:可迭代字典数组/类别集合
    """
    tree = ET.parse(filename)
    objects = []
    cls_set = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        cls_set.append(obj_struct['name'])
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects,set(cls_set)

def getDets(cls,objects,score=0,isRandom = True):
    dets = []
    for obj in objects:
        if cls == obj['name']:
            if isRandom:
                np.random.seed()
                score = np.random.random()
                print score
            det = obj['bbox']
            det.append(score)
            dets.append(det)
    # print dets
    return np.array(dets)

def getXmlDets(xml_dir,save_dir,detectColor,confidentThresh,ovr_thresh):
    for fileName in os.listdir(xml_dir):
        if re.match(".*.xml$", fileName):
            print("fileName = %s" % fileName)
            xml_path = os.path.join(xml_dir, fileName)
            objects, cls_set = parse_rec(xml_path)
            # print cls_set
            for cls in cls_set:
                dets = getDets(cls, objects)
                # print "{} = {}".format(cls,dets)
                imageName = fileName.split('.')[0]+'.jpg'
                plotXml(img_dir, imageName, save_dir, detectColor,cls, dets,"_xml_",confidentThresh,ovr_thresh)
                keep = py_cpu_nms(dets, ovr_thresh)
                plotXml(img_dir, imageName, save_dir, detectColor,cls, dets[keep,:], "_NMS_",confidentThresh,ovr_thresh)
                yield dets


def plotXml(img_dir, imgName,save_dir,detectColor,class_name,dets,NMSname,confidentThresh=0,ovr_thresh=0):
    inds = np.where(dets[:,-1]>=confidentThresh)[0]
    # print "inds = ",inds
    if len(inds)==0:
        return
    img = cv2.imread(os.path.join(img_dir, imgName))
    img = img[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')
    for i in inds:
        bbox = dets[i,:4]
        score = dets[i,-1]
        ax.add_patch(plt.Rectangle((int(bbox[0]), int(bbox[1])), int(bbox[2]) - int(bbox[0]),
                                   int(bbox[3]) - int(bbox[1]), fill=False,
                                   edgecolor=detectColor[class_name], linewidth=2))
        ax.text(int(bbox[0]), int(bbox[1]) - 2, u'{} {:.3f}'.format(u'平头自卸车',score),
                bbox=dict(facecolor=detectColor[class_name], alpha=0.5),
                fontsize=14, color='white',fontproperties=zhfont)
    ax.set_title((u'{} 检测目标概率'
                  'P({} | box) >= {:.1f}').format(u'平头自卸车',u'平头自卸车',confidentThresh),
                 fontsize=14,fontproperties=zhfont)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    output_dir = os.path.join(save_dir, imgName.split('.')[0] + NMSname+class_name+".jpg")
    plt.savefig(output_dir)
    img = cv2.imread(output_dir)
    gray = rgb2gray(img)
    cv2.imwrite(os.path.join(save_dir,imgName.split('.')[0] + NMSname+class_name+"_gray.jpg"),gray)


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

if __name__ == '__main__':
    xml_dir = r'/home/alex/Downloads/labelImg/demo/tmp'
    img_dir = r'/home/alex/Downloads/labelImg/demo/tmp'
    save_dir = r'/home/alex/Downloads/labelImg/demo/detectResult'
    detectColor = {'singleBridgeVan': 'chocolate', 'doubleBridgeVan': 'mediumturquoise', 'halfHangDragon': 'khaki',
                   'fullHangDragon': 'darkorchid', 'flatHeadDumper': 'hotpink', 'sharpHeadDumper': 'red',
                   'ellipticalTank': 'turquoise', 'squareTank': 'burlywood', 'mixerTank': 'lightskyblue', 'dog': 'royalblue',
                   'horse': 'orange', 'motorbike': 'grey', 'person': 'chartreuse', 'pottedplant': 'dodgerblue',
                   'sheep': 'purple', 'sofa': 'aqua', 'train': 'tan', 'tvmonitor': 'plum'}
    ovr_thresh = 0.1
    confidentThresh = 0.1
    dets_list = getXmlDets(xml_dir,save_dir,detectColor,confidentThresh,ovr_thresh)
    for dets in dets_list:
        # print "-----------keep---------------"
        keep = py_cpu_nms(dets,ovr_thresh)
        # print dets[keep]

