from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.predict import predict

import matplotlib.pyplot as plt
import os

# LOADING THE MODEL
model = vgg_unet(n_classes=6, input_height=416, input_width=608)

# TRAINING THE MODEL ON CUSTOM DATASET
history = model.train(
    batch_size=2,
    steps_per_epoch=10,
    n_classes=6,
    ignore_zero_class=False,
    train_images=os.path.dirname(os.getcwd()) + "/datasets/data/images_prepped_train/",
    train_annotations=os.path.dirname(os.getcwd()) + "/datasets/data/annotations_prepped_train/",
    checkpoints_path=os.path.dirname(os.getcwd()) + "/datasets/op/vgg_unet_1", epochs=100
)

model.save(os.path.dirname(os.getcwd()) + "/datasets/op/vgg_unet")


plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()