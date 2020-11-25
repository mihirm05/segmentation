from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.predict import predict
import matplotlib.pyplot as plt
import os
from keras.models import load_model

print(os.path.dirname(os.getcwd()) )


# LOADING THE MODEL 
model = vgg_unet(n_classes=5, input_height=416, input_width=608)
#print(model.summary())

# TRAINING THE MODEL ON CUSTOM DATASET
'''
model.train(
    train_images=os.path.dirname(os.getcwd())+"/datasets/data/images_prepped_train/",
    train_annotations=os.path.dirname(os.getcwd())+"/datasets/data/annotations_prepped_train/",
    checkpoints_path=os.path.dirname(os.getcwd())+"/tmp_new/vgg_unet_1", epochs=5
)
'''

#m = load_model(os.path.dirname(os.getcwd())+"/datasets/tmp/vgg_unet_1.1")
#print(m.summary())
# TESTING THE MODEL ON TEST IMAGES (#1)
out = predict(
    inp=os.path.dirname(os.getcwd()) + "/datasets/data/images_prepped_test/b1-09517_Clipped.jpg",
    out_fname=os.path.dirname(os.getcwd()) + "/datasets/tmp/outputput1.png",
    checkpoints_path = os.path.dirname(os.getcwd())+"/datasets/tmp/vgg_unet_1.1")
print('Done')


# TESTING THE MODEL ON TEST IMAGES (#2)
'''                                                 
o = model.predict_segmentation(
    inp=os.path.dirname(os.getcwd()) + "/datasets/data/images_prepped_test/b123-752_Clipped.jpg",
    out_fname=os.path.dirname(os.getcwd()) + "/tmp_new/out.png", overlay_img=True, show_legends=True,
    class_names=["Sky", "Tree", "Road", "Terrain"]
)

# evaluating the model 
print(model.evaluate_segmentation(inp_images_dir=os.path.dirname(os.getcwd())+"/datasets/data/images_prepped_test/", annotations_dir=os.path.dirname(os.getcwd())+"/datasets/data/annotations_prepped_test/"))
'''
