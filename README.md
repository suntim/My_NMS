# My_NMS



#!/usr/bin/env python

import argparse
import json
import os
import os.path as osp
import warnings
import re
import numpy as np
import PIL.Image
import yaml

import utils
def pascal_palette():
  palette = {0 :(  0,   0,   0) ,
             1 :(128,   0,   0) ,
             2 :(  0, 128,   0) ,
             3 :(128, 128,   0) ,
             4 :(  0,   0, 128) ,
             5 :(128,   0, 128) ,
             6 :(  0, 128, 128) ,
             7 :(128, 128, 128) ,
             8 :( 64,   0,   0) ,
             9 :(192,   0,   0) ,
             10 :( 64, 128,   0) ,
             11 :(192, 128,   0) ,
             12:( 64,   0, 128) ,
             13:(192,   0, 128) ,
            14: ( 64, 128, 128) ,
            15: (192, 128, 128) ,
             16:(  0,  64,   0) ,
             17:(128,  64,   0) ,
             18:(  0, 192,   0) ,
             19:(128, 192,   0) ,
            20: (  0,  64, 128) }

  return palette
def extract_classes(segm):
    cl = np.unique(segm)#cls
    n_cl = len(cl)
    return cl, n_cl
def changeToDataset(json_dir,out_dir=None):


    # parser = argparse.ArgumentParser()
    # parser.add_argument('json_file')
    # parser.add_argument('-o', '--out', default=19949)
    # args = parser.parse_args()
    cls_dict = {}
    i=0
    for fileName in os.listdir(json_dir):
        if re.match(".*[.]json$",fileName):
            json_file = osp.join(json_dir,fileName)
            print (json_file)

            if out_dir is None:
                out_dir = osp.join(osp.dirname(json_file), "output")
            # else:
                # print 'Existed out_dir = ',out_dir

            if not osp.exists(out_dir):
                os.mkdir(out_dir)

            out_dir_label = os.path.join(out_dir, 'label')
            if not osp.exists(out_dir_label):
                os.mkdir(out_dir_label)

            out_dir_viz = os.path.join(out_dir,'viz')
            if not osp.exists(out_dir_viz):
                os.mkdir(out_dir_viz)

            # print ("out_dir_label = ",out_dir_label)
            print(osp.join(out_dir, fileName))

            data = json.load(open(json_file))
            img = utils.img_b64_to_array(data['imageData'])
            pred_image, lbl_names,clsDict = utils.labelme_shapes_to_label(img.shape, data['shapes'])
            for k in clsDict:
                if k not in cls_dict.values():
                    cls_dict[i]=k
                    i+=1
            # print(cls_dict)
            # cls_dict = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
            #             7: 'car', 8: 'cat',9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
            #             16: 'plant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv'}
            cl, n_cl = extract_classes(pred_image)
            captions = ['%d: %s' % (cl_index, cls_dict[cl_index]) for cl_index in cl]
            print("captions=",captions)
            # print("max(pred_image)=",np.max(pred_image))
            lbl_viz = utils.draw_label(pred_image, img, captions)

            #New Name
            NewName = fileName.split('.')[0]
            # print (osp.join(out_dir,NewName+ '.png'))
            # PIL.Image.fromarray(img).save(osp.join(out_dir,NewName+ '_img.png'))
            PIL.Image.fromarray(pred_image).save(osp.join(out_dir_label, NewName+ '.png'))
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir_viz, NewName+ '_label_viz.png'))

            with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                for lbl_name in lbl_names:
                    f.write(lbl_name + '\n')

            # warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=lbl_names)
            with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            # print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    json_dir = r'D:\Bill_Test\train'
    # out_dir = r'D:\Bill_Test\train'
    changeToDataset(json_dir)
	
	
import base64
import cv2
try:
    import io
except ImportError:
    import io as io
import warnings

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw


def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7-j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7-j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7-j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap


def labelcolormap(N=256):
    warnings.warn('labelcolormap is deprecated. Please use label_colormap.')
    return label_colormap(N=N)


# similar function as skimage.color.label2rgb
def label2rgb(lbl, img=None, n_labels=None, alpha=0.3, thresh_suppress=0):
    if n_labels is None:
        n_labels = len(np.unique(lbl))

    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        # img_gray = PIL.Image.fromarray(img).convert('LA')
        # img_gray = np.asarray(img_gray.convert('RGB'))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)
    return lbl_viz


def img_b64_to_array(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr


def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def draw_label(label, img, label_names, colormap=None):
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if colormap is None:
        max_index_lab = np.max(label) + 1
        colormap = label_colormap(max_index_lab)

    label_viz = label2rgb(label, img, n_labels=max_index_lab)
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_name in label_names:
        cl_index = int(label_name.split(':')[0])
        print("cl_index = ", cl_index)
        fc = colormap[cl_index]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append(label_name)
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)

    f = io.BytesIO()
    plt.savefig(f, bbox_inches='tight', pad_inches=0)
    plt.cla()
    plt.close()

    out_size = (img.shape[1], img.shape[0])
    out = PIL.Image.open(f).resize(out_size, PIL.Image.BILINEAR).convert('RGB')
    out = np.asarray(out)
    return out


def labelme_shapes_to_label(img_shape, shapes):
    label_name_to_val = {'background': 0}
    lbl = np.zeros(img_shape[:2], dtype=np.int32)
    for shape in shapes:
        polygons = shape['points']
        label_name = shape['label']
        if label_name in label_name_to_val:
            label_value = label_name_to_val[label_name]
        else:
            label_value = len(label_name_to_val)
            label_name_to_val[label_name] = label_value
        mask = polygons_to_mask(img_shape[:2], polygons)
        lbl[mask] = label_value

    lbl_names = [None] * (max(label_name_to_val.values()) + 1)
    for label_name, label_value in label_name_to_val.items():
        lbl_names[label_value] = label_name

    return lbl, lbl_names,label_name_to_val

	# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import matplotlib
matplotlib.use('pdf')
import tensorflow as tf
import numpy as np
from scipy.misc import imread
from matplotlib import pyplot as plt
import network
slim = tf.contrib.slim
import os,re,cv2
import argparse
import json
import PIL.Image
from metrics import *
try:
    import io
except ImportError:
    import io as io

plt.interactive(False)

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Eval params')
envarg.add_argument("--model_id", type=int, help="Model id name to be loaded.")
envarg.add_argument("--gpu_id", type=int, help="Gpu id name to be used.")
envarg.add_argument("--batch_size", type=int, default = 10, help="Batch size of test.")
envarg.add_argument("--ResizeWidth", type=int, default = 513, help="Resize Width of pic.")
envarg.add_argument("--ResizeHeight", type=int, default = 513, help="Resize Height of pic.")
envarg.add_argument("--save_Result", type=bool, default = False, help="Want to save result?.")
envarg.add_argument("--Pic_Dir", help="Test Pic_Dir.")
input_args = parser.parse_args()

# best: 16645
model_name = str(input_args.model_id)

# uncomment and set the GPU id if applicable.
os.environ["CUDA_VISIBLE_DEVICES"]=str(input_args.gpu_id)

log_folder = './tboard_logs'

if not os.path.exists(os.path.join(log_folder, model_name, "test")):
    os.makedirs(os.path.join(log_folder, model_name, "test"))

with open(log_folder + '/' + model_name + '/train/data.json', 'r') as fp:
    args = json.load(fp)

class Dotdict(dict):
     """dot.notation access to dictionary attributes"""
     __getattr__ = dict.get
     __setattr__ = dict.__setitem__
     __delattr__ = dict.__delitem__

args = Dotdict(args)

class_labels = [v for v in range((args.number_of_classes+1))]
class_labels[-1] = 255

def scale_image_with_crop_padding(image, shapes):
    image_croped = tf.image.resize_image_with_crop_or_pad(image,input_args.ResizeHeight,input_args.ResizeWidth)

    # Shift all the classes by one -- to be able to differentiate
    # between zeros representing padded values and zeros representing
    # a particular semantic class.
    return image_croped, shapes

def _Resize_function(image_decoded,shapes):
    # image_string = tf.read_file(filename)
    # image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_images(image_decoded, [input_args.ResizeWidth,input_args.ResizeHeight])
    print("image_resized = ",image_resized)
    return image_resized,shapes

def tf_record_parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
        "height": tf.FixedLenFeature((), tf.int64),
        "width": tf.FixedLenFeature((), tf.int64)
    }
    features = tf.parse_single_example(record, keys_to_features)

    image = tf.decode_raw(features['image_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    # reshape input and annotation images
    image = tf.reshape(image, (height, width, 3), name="image_reshape")
    return tf.to_float(image), (height, width)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tfrecord_dataset(Test_images_dir,filename_list, writer):
    # create training tfrecord
    for i, image_name in enumerate(filename_list):
        try:
            image_np = imread(os.path.join(Test_images_dir, image_name.strip()))
        except FileNotFoundError:
            # read from Pascal VOC path
            print("{} not Exited!!".format(os.path.join(Test_images_dir, image_name.strip())))


        image_h = image_np.shape[0]
        image_w = image_np.shape[1]

        img_raw = image_np.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(image_h),
            'width': _int64_feature(image_w),
            'image_raw': _bytes_feature(img_raw)}))

        writer.write(example.SerializeToString())

    print("End of TfRecord. Total of image written:", i+1)
    writer.close()

def label_colormap(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7-j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7-j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7-j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

# similar function as skimage.color.label2rgb
def label2rgb(lbl, img=None, n_labels=None, alpha=0.5, thresh_suppress=0):
    if n_labels is None:
        n_labels = len(np.unique(lbl))

    cmap = label_colormap(n_labels)
    cmap = (cmap * 255).astype(np.uint8)

    lbl_viz = cmap[lbl]
    lbl_viz[lbl == -1] = (0, 0, 0)  # unlabeled

    if img is not None:
        # img_gray = PIL.Image.fromarray(img).convert('LA')
        # img_gray = np.asarray(img_gray.convert('RGB'))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        lbl_viz = alpha * lbl_viz + (1 - alpha) * img_gray
        lbl_viz = lbl_viz.astype(np.uint8)

    return lbl_viz

def draw_label(label, img, label_names,save_path,colormap=None):
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0,
                        wspace=0, hspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    if colormap is None:
        max_index_lab = np.max(label)+1
        colormap = label_colormap(max_index_lab)
        print("np.max(label)+1=", max_index_lab)

    label_viz = label2rgb(label, img, n_labels=max_index_lab)
    plt.imshow(label_viz)
    plt.axis('off')

    plt_handlers = []
    plt_titles = []
    for label_name in label_names:
        cl_index = int(label_name.split(':')[0])
        print("cl_index = ",cl_index)
        fc = colormap[cl_index]
        p = plt.Rectangle((0, 0), 1, 1, fc=fc)
        plt_handlers.append(p)
        plt_titles.append(label_name)
    plt.legend(plt_handlers, plt_titles, loc='lower right', framealpha=.5)
    plt.savefig(save_path, dpi = 400, bbox_inches = "tight")

def extract_masks(segm, cl, n_cl):
    h, w  = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c
    return masks

def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width

def extract_classes(segm):
    cl = np.unique(segm)#cls
    n_cl = len(cl)
    return cl, n_cl

if __name__ == '__main__':
    Test_images_dir = input_args.Pic_Dir  # 图片地址
    test_images_filename_list = []
    for filName in os.listdir(Test_images_dir):
        if re.match(".*[.]jpg",filName):
            test_images_filename_list.append(filName)

    # print("test_images_filename_list = ",test_images_filename_list)
    test_filenames = os.path.join(Test_images_dir,'test.tfrecords')
    test_writer = tf.python_io.TFRecordWriter(test_filenames)
    create_tfrecord_dataset(Test_images_dir, test_images_filename_list, test_writer)
    test_dataset = tf.data.TFRecordDataset([test_filenames])
    test_dataset = test_dataset.map(tf_record_parser)  # Parse the record into tensors.
    test_dataset = test_dataset.map(_Resize_function)
    # test_dataset = test_dataset.map(scale_image_with_crop_padding)
    test_dataset = test_dataset.batch(input_args.batch_size)

    iterator = test_dataset.make_one_shot_iterator()
    batch_images_tf, batch_shapes_tf = iterator.get_next()

    logits_tf = network.deeplab_v3(batch_images_tf, args, is_training=False, reuse=False)


    predictions_tf = tf.argmax(logits_tf, axis=3)
    # probabilities_tf = tf.nn.softmax(logits_tf)
    saver = tf.train.Saver()

    # test_folder = test_filenames
    train_folder = os.path.join(log_folder, model_name, "train")

    with tf.Session() as sess:
        index = 1
        # Create a saver.
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        saver.restore(sess, os.path.join(train_folder, "model.ckpt"))
        print("Model", model_name, "restored.")

        while True:
            try:
                batch_images_np, batch_predictions_np, batch_shapes_np = sess.run(
                    [batch_images_tf, predictions_tf, batch_shapes_tf])
                heights, widths = batch_shapes_np

                # loop through the images in the batch and extract the valid areas from the tensors
                for i in range(batch_predictions_np.shape[0]):
                    print("index/batch_predictions_np.shape[0] = {}/{}".format(index, batch_predictions_np.shape[0]))

                    pred_image = batch_predictions_np[i]
                    input_image = batch_images_np[i]

                    # remove scale_image_with_crop_padding == 255
                    # indices = np.where(pred_image != 255)
                    # pred_image = pred_image[indices]
                    # input_image = input_image[indices]

                    print("pred_image.shape[0]*pred_image[1] = {}*{}={}".format(pred_image.shape[0],pred_image.shape[1],pred_image.shape[0]*pred_image.shape[1]))
                    print("input_image.shape[0] = ",input_image.shape[0])
                    sizeofShape = pred_image.shape[0]*pred_image.shape[1]
                    if sizeofShape == input_args.ResizeWidth * input_args.ResizeHeight:
                        pred_image = np.reshape(pred_image, (input_args.ResizeWidth,input_args.ResizeHeight))
                        input_image = np.reshape(input_image, (input_args.ResizeWidth,input_args.ResizeHeight, 3))
                    else:
                        pred_image = np.reshape(pred_image, (heights[i], widths[i]))
                        input_image = np.reshape(input_image, (heights[i], widths[i], 3))

                    if (input_args.save_Result):
                        cls_dict = {0:'background',1:'aeroplane',2:'bicycle',3:'bird',4:'boat',5:'bottle',6:'bus',7:'car',8:'cat',
                                    9:'chair',10:'cow',11:'diningtable',12:'dog',13:'horse',14:'motorbike',15:'person',
                                    16:'plant',17:'sheep',18:'sofa',19:'train',20:'tv'}
                        cl, n_cl = extract_classes(pred_image)

                        captions = ['%d: %s' % (cl_index, cls_dict[cl_index]) for cl_index in cl]
                        print("pred_image.shape=", pred_image.shape)
                        print("input_image.shape=", input_image.shape)
                        print("captions=", captions)

                        save_dir = "./ouput"
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)

                        save_path = os.path.join(save_dir, "mask_" + str(index) + ".jpg")
                        draw_label(pred_image, input_image, captions,save_path)
                        index += 1
            except tf.errors.OutOfRangeError:
                break

