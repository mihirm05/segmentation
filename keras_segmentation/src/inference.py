from keras_segmentation.models.unet import vgg_unet
from IPython.display import Image
import cv2
import os

import matplotlib.pyplot as plt

from IPython.display import Image

model = vgg_unet(n_classes=51, input_height=320, input_width=640)

model.train(
    n_classes=5,
    batch_size=64,
    train_images=os.path.dirname(os.getcwd()) + "/datasets/dataDemo/images_prepped_train/",
    train_annotations=os.path.dirname(os.getcwd()) + "/datasets/dataDemo/annotations_prepped_train/",
    checkpoints_path=os.path.dirname(os.getcwd()) + "/output/vgg_unet_1", epochs=1
)

# out = model.predict_segmentation(
#    inp=os.path.dirname(os.getcwd()) + "/datasets/data/images_prepped_test/b45-0046_Clipped.jpg",
#    out_fname=os.path.dirname(os.getcwd()) + "/datasets/data/tmp_data/outputput.png"
# )

o = model.predict_segmentation(
    inp=os.path.dirname(os.getcwd()) + "/datasets/dataDemo/IMG_9894.JPG",
    out_fname=os.path.dirname(os.getcwd()) + "/datasets/tmp_dataDemo/outputput.png", overlay_img=True, show_legends=True,
    class_names=["Sky", "Grass", "Tree", "Terrain"],
)ResNet
plt.imshow(o)
Image(os.path.dirname(os.getcwd()) + "/datasets/tmp_dataDemo/outputput.png")
print('DONE')
#
