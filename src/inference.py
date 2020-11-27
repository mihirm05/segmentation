import cv2
import os
import matplotlib.pyplot as plt

from keras.models import load_model
from keras_segmentation.models.unet import vgg_unet
from IPython.display import Image
from keras_segmentation.metrics import get_iou
from keras_segmentation.models.model_utils import transfer_weights
from IPython.display import Image
from keras_segmentation.predict import predict_multiple

model = vgg_unet(n_classes=6, input_height=320, input_width=640)
m = load_model(os.path.dirname(os.getcwd()) + "/datasets/op/vgg_unet")

transfer_weights(m, model, verbose=True)
'''
# TESTING THE MODEL ON TEST IMAGES (#1)
out = model.predict_segmentation(
    inp=os.path.dirname(os.getcwd()) + "/datasets/data/images_prepped_test/b1-09517_Clipped.jpg",
    out_fname=os.path.dirname(os.getcwd()) + "/datasets/op/outout1.png",
    checkpoints_path=os.path.dirname(os.getcwd()) + "/datasets/op/vgg_unet_1")
print('Done')

# TESTING THE MODEL ON TEST IMAGES (#2)
o = model.predict_segmentation(
    inp=os.path.dirname(os.getcwd()) + "/datasets/data/images_prepped_test/b1-09517_Clipped.jpg",
    out_fname=os.path.dirname(os.getcwd()) + "/datasets/op/outout2.png", overlay_img=False, show_legends=True,
    class_names=["Sky", "Forest Floor", "Vegetation", "Grass", "Obstacle", "Tree"],
    checkpoints_path=os.path.dirname(os.getcwd()) + "/datasets/op/vgg_unet_1")

# TESTING THE MODEL ON TEST IMAGES (#2)
o = model.predict_segmentation(
    inp=os.path.dirname(os.getcwd()) + "/datasets/data/images_prepped_test/b1-09517_Clipped.jpg",
    out_fname=os.path.dirname(os.getcwd()) + "/datasets/op/outout2TRUE.png", overlay_img=True, show_legends=True,
    class_names=["Sky", "Forest Floor", "Vegetation", "Grass", "Obstacle", "Tree"],
    checkpoints_path=os.path.dirname(os.getcwd()) + "/datasets/op/vgg_unet_1")
'''
# evaluating the model
# print(model.evaluate_segmentation(inp_images_dir=os.path.dirname(os.getcwd())+"/datasets/data/images_prepped_test/",
#                                  annotations_dir=os.path.dirname(os.getcwd())+"/datasets/data/annotations_prepped_test/"))

'''
model.predict_multiple(checkpoints_path=os.path.dirname(os.getcwd()) + "/datasets/op/vgg_unet_1",
                 inp_dir=os.path.dirname(os.getcwd()) + "/datasets/data/images_prepped_test/",
                 out_dir=os.path.dirname(os.getcwd()) + "/datasets/outputpredictions",
                 overlay_img=True,
                 show_legends=True,
                 class_names=["Sky", "Forest Floor", "Vegetation", "Grass", "Obstacle", "Tree"])
'''
model.predict_multiple(checkpoints_path=os.path.dirname(os.getcwd()) + "/datasets/op/vgg_unet_1",
                       inp_dir=os.path.dirname(os.getcwd()) + "/datasets/data/images_prepped_test/",
                       out_dir=os.path.dirname(os.getcwd()) + "/datasets/outputpredictionswithoutoverlay",
                       show_legends=True,
                       class_names=["Sky", "Forest Floor", "Vegetation", "Grass", "Obstacle", "Tree"])
