from model import *
from model_crackdet_15 import *
from model_2i2o import *
import os.path
from cal_resize_iou import cal_resize_iou
import cv2 as cv
import imageio
import numpy as np
from skimage import transform as trans
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image

fullscale_img_dir = 'D:\\IPC_SHM2\\image_test\\'
# fullscale_label_dir = 'project2\\fullsize\\crack\\'

# --------------------------------------------------Test----------------------------------------------------

def batch_test(model_name, model, test_input_dir, test_save_dir):
    os.makedirs(test_save_dir+'component/', exist_ok=True)
    os.makedirs(test_save_dir + 'crack/', exist_ok=True)
    os.makedirs(test_save_dir + 'spall/', exist_ok=True)
    os.makedirs(test_save_dir + 'rebar/', exist_ok=True)
    os.makedirs(test_save_dir+'ds/', exist_ok=True)
    os.makedirs(test_save_dir+'colored_component/', exist_ok=True)
    os.makedirs(test_save_dir+'colored_crack/', exist_ok=True)
    os.makedirs(test_save_dir+'colored_spall/', exist_ok=True)
    os.makedirs(test_save_dir+'colored_rebar/', exist_ok=True)
    os.makedirs(test_save_dir+'colored_ds/', exist_ok=True)
    model.load_weights(model_name + '.hdf5')
    data_name = os.listdir(test_input_dir)
    threshold=0.5
    for i in data_name:
        img = Image.open(os.path.join(test_input_dir, i))
        img = img.resize((512, 512), Image.NEAREST)
        img = np.array(img)
        img = img/255
        color_map1 = [[202, 150, 150], [198, 186, 100], [167, 183, 186], [255, 255, 133], [192, 192, 206], [32, 80, 160],
                     [193, 134, 1], [70, 70, 70]]
        color_map2 = [[0, 255, 0], [150, 250, 0], [255, 225, 50], [255, 0, 0], [128, 128, 128]]
        img = np.reshape(img, (1,) + img.shape)
        results = model.predict(img, verbose=1)
        results1 = [np.squeeze(results[0]), np.squeeze(results[4])]

        predict1 = trans.resize(results1[0],(1080,1920),clip=True,preserve_range=True)
        predict5 = trans.resize(results1[1],(1080,1920),clip=True,preserve_range=True)
        predict1 = predict1.argsort(2)[:, :, -1]
        predict5 = predict5.argsort(2)[:, :, -1]
        predict1 = predict1.astype(np.uint8)
        predict5 = predict5.astype(np.uint8)

        paint1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        paint2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for j in range(8):
            paint1[predict1 == j] = color_map1[j]
        for k in range(5):
            paint2[predict5 == k] = color_map2[k]
        predict1 = predict1.astype(np.uint8)
        predict5 = predict5.astype(np.uint8)
        filename = i.split('.')[0]
        imageio.imsave(os.path.join(test_save_dir+'component/', filename + ".png"), predict1)
        imageio.imsave(os.path.join(test_save_dir+'ds/', filename + ".png"), predict5)
        imageio.imsave(os.path.join(test_save_dir+'colored_component/', filename + ".png"), paint1)
        imageio.imsave(os.path.join(test_save_dir+'colored_ds/', filename + ".png"), paint2)
        results2 = np.squeeze(results[1])
        results3 = np.squeeze(results[2])
        results4 = np.squeeze(results[3])

        results2 = trans.resize(results2,(1080,1920),clip=True,preserve_range=True)
        results3 = trans.resize(results3,(1080,1920),clip=True,preserve_range=True)
        results4 = trans.resize(results4,(1080,1920),clip=True,preserve_range=True)

        results2[results2 <= threshold] = 0
        results2[results2 > threshold] = 1
        results3[results3 <= threshold] = 0
        results3[results3 > threshold] = 1
        results4[results4 <= threshold] = 0
        results4[results4 > threshold] = 1

        imageio.imsave(os.path.join(test_save_dir+'colored_crack/', filename + ".png"), results2)
        imageio.imsave(os.path.join(test_save_dir+'colored_spall/', filename + ".png"), results3)
        imageio.imsave(os.path.join(test_save_dir+'colored_rebar/', filename + ".png"), results4)
        results2 = results2.astype(np.uint8)
        results3 = results3.astype(np.uint8)
        results4 = results4.astype(np.uint8)

        imageio.imsave(os.path.join(test_save_dir+'crack/', filename + ".png"), results2)
        imageio.imsave(os.path.join(test_save_dir+'spall/', filename + ".png"), results3)
        imageio.imsave(os.path.join(test_save_dir+'rebar/', filename + ".png"), results4)

   #     imageio.imsave(os.path.join(colored_test_save_dir1, filename + ".png"), mask1)
     #   imageio.imsave(os.path.join(colored_test_save_dir2, filename + ".png"), mask2)


def eval(source_dir, save_dir, model_name, model, label_dir =None):
    """
    This is to enable end-to-end evaluation of full-scale images with size of 1920*1080, save predicted masks and
    output indicators including IoU, accuracy, precision, recall, and F1 score.
    :param
    source_dir: the directory of the test fullscale images
    save_dir: the root save path
    model_name: weight to be loaded
    model: structure of the model to be tested
    :return
    The indicator array of [IoU, accuracy, precision, recall, F1 score]
    Predicted masks, which are saved within the root save path.
    """
    save_dir = save_dir + model_name + '/'
    os.makedirs(save_dir, exist_ok=True)
   # crop_img_dir = save_dir+'/crop_img/'
  #  os.makedirs(crop_img_dir, exist_ok=True)

    # read each single full-scale image
  #  crop(source_dir, crop_img_dir)

    batch_test(model_name, model, fullscale_img_dir, test_save_dir=save_dir)

  #  joint_resize(save_dir+'crop_mask/', save_dir+'full_mask/')

    # cal_resize_iou(save_dir+'full_mask/', label_dir, save_dir, model_name)


if __name__ == '__main__':
       eval(fullscale_img_dir, save_dir='project2/test results/',
            model_name='Selfnet_P2_81115V11_wfaug_512size', model=newmodel_noname())


