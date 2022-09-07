from model import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def adjustData(img,mask,mask1,mask2,mask3,mask4,flag_multi_class,num_class,num_class1,num_class2,num_class3,num_class4):
    if (flag_multi_class):
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        mask1 = mask1[:, :, :, 0] if (len(mask1.shape) == 4) else mask1[:, :, 0]
        mask2 = mask2[:, :, :, 0] if (len(mask2.shape) == 4) else mask2[:, :, 0]
        mask3 = mask3[:, :, :, 0] if (len(mask3.shape) == 4) else mask3[:, :, 0]
        mask4 = mask4[:, :, :, 0] if (len(mask4.shape) == 4) else mask4[:, :, 0]



        new_mask = np.zeros(mask.shape + (num_class,))
        new_mask1 = np.zeros(mask1.shape + (num_class1,))
        new_mask2 = np.zeros(mask2.shape + (num_class2,))
        new_mask3 = np.zeros(mask3.shape + (num_class3,))
        new_mask4 = np.zeros(mask4.shape + (num_class4,))

        new_mask[mask == 0, 0] = 1
        new_mask[mask == 1, 1] = 1
        new_mask[mask == 2, 2] = 1
        new_mask[mask == 3, 3] = 1
        new_mask[mask == 4, 4] = 1
        new_mask[mask == 5, 5] = 1
        new_mask[mask == 6, 6] = 1
        new_mask[mask == 100, 7] = 1

        new_mask1[mask1 == 1, 0] = 1


        new_mask2[mask2 == 1, 0] = 1


        new_mask3[mask3 == 1, 0] = 1


        new_mask4[mask4 == 0, 0] = 1
        new_mask4[mask4 == 1, 1] = 1
        new_mask4[mask4 == 2, 2] = 1
        new_mask4[mask4 == 3, 3] = 1
        new_mask4[mask4 == 100, 4] = 1



        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] ,new_mask.shape[2],
                                         new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
        new_mask.shape[0] , new_mask.shape[1], new_mask.shape[2]))
        new_mask1 = np.reshape(new_mask1, (new_mask1.shape[0], new_mask1.shape[1] ,new_mask1.shape[2],
                                         new_mask1.shape[3])) if flag_multi_class else np.reshape(new_mask1, (
        new_mask1.shape[0] , new_mask1.shape[1], new_mask1.shape[2]))
        new_mask2 = np.reshape(new_mask2, (new_mask2.shape[0], new_mask2.shape[1] ,new_mask2.shape[2],
                                         new_mask2.shape[3])) if flag_multi_class else np.reshape(new_mask2, (
        new_mask2.shape[0] , new_mask2.shape[1], new_mask2.shape[2]))
        new_mask3 = np.reshape(new_mask3, (new_mask3.shape[0], new_mask3.shape[1] ,new_mask3.shape[2],
                                         new_mask3.shape[3])) if flag_multi_class else np.reshape(new_mask3, (
        new_mask3.shape[0] , new_mask3.shape[1], new_mask3.shape[2]))
        new_mask4 = np.reshape(new_mask4, (new_mask4.shape[0], new_mask4.shape[1] ,new_mask4.shape[2],
                                         new_mask4.shape[3])) if flag_multi_class else np.reshape(new_mask4, (
        new_mask4.shape[0] , new_mask4.shape[1], new_mask4.shape[2]))

        mask = new_mask
        mask1 = new_mask1
        mask2 = new_mask2
        mask3 = new_mask3
        mask4 = new_mask4

    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

        mask1 = mask1 /255
        mask1[mask1 > 0.5] = 1
        mask1[mask1 <= 0.5] = 0
    return (img,mask,mask1,mask2,mask3,mask4)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,mask1_folder,mask2_folder,mask3_folder,mask4_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = True,num_class = 8,num_class1 = 1,num_class2 = 1,num_class3 = 1,num_class4 = 5,save_to_dir = None,target_size = (512,512),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None ,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    mask1_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask1_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    mask2_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask2_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    mask3_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask3_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    mask4_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask4_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_generator = zip(image_generator, mask_generator,mask1_generator,mask2_generator,mask3_generator,mask4_generator)

    for (img,mask,mask1,mask2,mask3,mask4) in train_generator:
         img,mask,mask1,mask2,mask3,mask4 = adjustData(img,mask,mask1,mask2,mask3,mask4,flag_multi_class,num_class,num_class1,num_class2,num_class3,num_class4)
         yield (img,[mask,mask1,mask2,mask3,mask4])

#image augmentation
data_gen_args = dict(horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2, 'project2', image_folder='image', mask_folder='component', mask1_folder='crack',mask2_folder='spall',mask3_folder='rebar',mask4_folder='ds',aug_dict=data_gen_args, save_to_dir = None)
print("loading training data done")
myGene_valid = trainGenerator(1, 'project2', image_folder='image', mask_folder='component', mask1_folder='crack',mask2_folder='spall',mask3_folder='rebar',mask4_folder='ds',aug_dict=data_gen_args, save_to_dir = None)
print("loading validation data done")
model_name = 'selfnet'
print("got unet")
model = newmodel_noname()
Early_Stopping=EarlyStopping(monitor='val_loss', patience=5,  mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, mode='auto')
model_checkpoint = ModelCheckpoint(model_name+".hdf5", monitor='val_loss', verbose=1, save_best_only=True)

print('Fitting model...')
history = model.fit_generator(myGene,steps_per_epoch=2000, epochs=80,validation_data=myGene_valid,validation_steps=200,callbacks=[model_checkpoint,Early_Stopping,reduce_lr])


with open(model_name+".txt", 'w') as f:
    f.write(str(history.history))


print('model saved ...{}'.format(model_name+".hdf5"))
acc = history.history['iou_score']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

plt.title('Accuracy and Loss')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, loss, 'blue', label='Validation loss')
plt.legend()
plt.show()
