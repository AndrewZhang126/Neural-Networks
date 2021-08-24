# The following code was adapted from Week 3 Programming Assignment 1 in the Convolutional Neural Networks course by DeepLearning.AI offered on Coursera
# https://www.coursera.org/learn/convolutional-neural-networks/home/week/3



import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from tensorflow.keras.models import load_model

#pre-trained YOLO model from https://github.com/allanzelener/YAD2K
from yad2k.models.keras_yolo import yolo_head
from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image

get_ipython().run_line_magic('matplotlib', 'inline')



"""
This function filters YOLO boxes according to a threshold value
"""
def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold = .6):

    # compute box scores
    box_scores = box_confidence * box_class_probs

    # find the box class and score
    box_classes = tf.math.argmax(box_scores, axis=-1)
    box_class_scores = tf.math.reduce_max(box_scores, axis=-1)
    
    # create a filtering mask that keeps track of which boxes to keep based on the threshold
    filtering_mask = box_class_scores >= threshold
    
    # apply the mask 
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes



"""
This function implements the Intersection over Union between two boxes
"""
def iou(box1, box2):
    
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    # calculate the area of the intersection
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = xi2 - xi1
    inter_height =  yi2 - yi1
    inter_area = max(inter_height, 0) * max(inter_width, 0)
    
    # calculate the union area according to the formula: A + B - Intersection(A,B)
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area
    
    # compute the Intersection over Union
    iou = inter_area / union_area
    
    return iou



"""
This function implements non-max suppression on a set of boxes
"""
def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    
    # max_boxes is the maximum number of boxes you would like
    
    max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')     

    # get the boxes you want to keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold) 

    # select the boxes to keep
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes


"""
This function converts YOLO boxes to bounding box corners
"""
def yolo_boxes_to_corners(box_xy, box_wh):
    
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return tf.keras.backend.concatenate([
        box_mins[..., 1:2],  
        box_mins[..., 0:1],  
        box_maxes[..., 1:2],  
        box_maxes[..., 0:1]  
    ])



"""
This function converts the output boxes of YOLO to the desired boxes
"""
def yolo_eval(yolo_outputs, image_shape = (720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    
    # yolo_outputs contains all the boxes the YOLO model predicts
    # image_shape is the input shape
    # max_boxes is the maximum number of predicted boxes
    # score_threshold is used for filtering boxes
    #iou_threshold is used for non-max supression filtering
    
    # retrieve the size, confidence, and class probabilities for all boxes predicted
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    
    # convert box sizes to corner coordinates for filtering
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    # helper function filters boxes by removing those below the score threshold
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, score_threshold)
    
    # scale boxes back to original image shape
    boxes = scale_boxes(boxes, image_shape)
    
    # helper function performs non-max supression to remove overlapping boxes that violate an IoU threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    return scores, boxes, classes



# contains data about the classes and boxes
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
model_image_size = (608, 608) 



# load pre-trained Keras YOLO model
yolo_model = load_model("model_data/", compile=False)



"""
This function preprocesses an image
"""
def preprocess_image(img_path, model_image_size):
    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) 
    return image, image_data



"""
This function generates random colors for all classes
"""
def get_colors_for_classes(num_classes):
    
    # use previously generated colors if same number of classes
    if (hasattr(get_colors_for_classes, "colors") and
            len(get_colors_for_classes.colors) == num_classes):
        return get_colors_for_classes.colors

    hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  
    random.shuffle(colors) 
    random.seed(None) 
    get_colors_for_classes.colors = colors  
    return colors



"""
This function draws bounding boxes on the image
"""
def draw_boxes(image, boxes, box_classes, class_names, scores=None):
    font = ImageFont.truetype(
        font='font/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    colors = get_colors_for_classes(len(class_names))
    for i, c in list(enumerate(box_classes)):
        box_class = class_names[c]
        box = boxes[i]
        
        if isinstance(scores.numpy(), np.ndarray):
            score = scores.numpy()[i]
            label = '{} {:.2f}'.format(box_class, score)
        else:
            label = '{}'.format(box_class)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return np.array(image)



"""
This function prints and plots the prediction boxes for an image
"""
def predict(image_file):
    
    # image_file is the name of the image (should be stored in an "images" folder)

    # preprocess the image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))
    
    yolo_model_outputs = yolo_model(image_data)
    yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    
    out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1],  image.size[0]], 10, 0.3, 0.5)

    # print predictions 
    print('Found {} boxes for {}'.format(len(out_boxes), "images/" + image_file))
    
    # generate colors for drawing bounding boxes.
    colors = get_colors_for_classes(len(class_names))
    
    # draw bounding boxes on the image file
    draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    
    # save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=100)
    
    # display the results
    output_image = Image.open(os.path.join("out", image_file))
    imshow(output_image)

    return out_scores, out_boxes, out_classes



# References
# [1] https://www.coursera.org/learn/convolutional-neural-networks/programming/3VCFG/car-detection-with-yolo
# [2] https://arxiv.org/abs/1506.02640
# [3] https://arxiv.org/abs/1612.08242
# [4] https://github.com/allanzelener/YAD2K
# [5] https://pjreddie.com/darknet/yolo/
