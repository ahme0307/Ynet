
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,BatchNormalization,Activation,UpSampling2D,Flatten, Dense,AveragePooling2D,add,AveragePooling2D,add,Dropout,ZeroPadding2D,Convolution2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy
from params import *
from keras.models import Sequential
import cv2
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from keras.utils import get_file
from keras import layers
from keras import utils
from keras import engine

#weights_path="enco_weights4.hdf5"
weights_path='vgg19_weights_tf_dim_ordering_tf_kernels.h5'
Ynetweigh='Y10_net.hdf5'
LR_mult_dict = {}
reduce=0.01
smooth=0.0001
LR_mult_dict['block1_conv1']=reduce
LR_mult_dict['block1_conv2']=reduce

LR_mult_dict['block2_conv1']=reduce
LR_mult_dict['block2_conv2']=reduce

LR_mult_dict['block3_conv1']=reduce
LR_mult_dict['block3_conv2']=reduce
LR_mult_dict['block3_conv3']=reduce
LR_mult_dict['block3_conv4']=reduce


LR_mult_dict['block4_conv1']=reduce
LR_mult_dict['block4_conv2']=reduce
LR_mult_dict['block4_conv3']=reduce
LR_mult_dict['block4_conv4']=reduce

LR_mult_dict['block5_conv1']=reduce
LR_mult_dict['block5_conv2']=reduce
LR_mult_dict['block5_conv3']=reduce
LR_mult_dict['block5_conv4']=reduce



def Y_net():
    include_top=False
    pooling=None
    classes=1000
    num_classes=1
    img_input=Input((None, None, 3))
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    down0aT = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(down0aT)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    down0T = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(down0T)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    down1T = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(down1T)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    down2T = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(down2T)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    down3T = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    xold = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(down3T)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(xold)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(xold)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)


    # Create model.
    model = Model(img_input, x, name='vgg19')


    #model.load_weights(weights_path)
        # 16  # center
    y = Conv2D(64, (3, 3), activation='selu', padding='same', name='BELblock1_conv1')(img_input)
    down0aB = Conv2D(64, (3, 3), activation='relu', padding='same', name='BELblock1_conv2')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='BELblock1_pool')(down0aB)

    # Block 2
    y = Conv2D(128, (3, 3), activation='selu', padding='same', name='BELblock2_conv1')(y)
    down0B = Conv2D(128, (3, 3), activation='relu', padding='same', name='BELblock2_conv2')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='BELblock2_pool')(down0B)

    # Block 3
    y = Conv2D(256, (3, 3), activation='selu', padding='same', name='BELblock3_conv1')(y)
    y = Conv2D(256, (3, 3), activation='selu', padding='same', name='BELblock3_conv2')(y)
    y = Conv2D(256, (3, 3), activation='selu', padding='same', name='BELblock3_conv3')(y)
    down1B = Conv2D(256, (3, 3), activation='selu', padding='same', name='BELblock3_conv4')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='BELblock3_pool')(down1B)

    # Block 4
    y = Conv2D(512, (3, 3), activation='selu', padding='same', name='BELblock4_conv1')(y)
    y = Conv2D(512, (3, 3), activation='selu', padding='same', name='BELblock4_conv2')(y)
    y = Conv2D(512, (3, 3), activation='selu', padding='same', name='BELblock4_conv3')(y)
    down2B = Conv2D(512, (3, 3), activation='selu', padding='same', name='BELblock4_conv4')(y)
    y = MaxPooling2D((2, 2), strides=(2, 2), name='BELblock4_pool')(down2B)

    # Block 5
    y = Conv2D(512, (3, 3), activation='selu', padding='same', name='BELblock5_conv1')(y)
    y = Conv2D(512, (3, 3), activation='selu', padding='same', name='BELblock5_conv2')(y)
    y = Conv2D(512, (3, 3), activation='selu', padding='same', name='BELblock5_conv3')(y)
    down3B = Conv2D(512, (3, 3), activation='relu', padding='same', name='BELblock5_conv4')(y)
    ynew = MaxPooling2D((2, 2), strides=(2, 2), name='BEblock5_pool')(down3B)
    
    
    if include_top:
        # Classification block
        y = Flatten(name='flatten')(ynew)
        y = Dense(4096, activation='relu', name='fc1')(y)
        y = Dense(4096, activation='relu', name='fc2')(y)
        y = Dense(classes, activation='softmax', name='predictions')(y)
    else:
        if pooling == 'avg':
            y = GlobalAveragePooling2D()(ynew)
        elif pooling == 'max':
            y = GlobalMaxPooling2D()(y)


    # Create model.
    model = Model(img_input, y, name='vgg19')


    #model.load_weights(weights_path)
    
    
    
    #center = add([xold, ynew])
    center = concatenate([xold, ynew], axis=3)
    center=conv_block(center, 1024, 7, 'center', strides=(2, 2))
    center=conv_block(center, 1024, 7, 'center_2', strides=(2, 2))
    # 32
    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([concatenate([down3T,down3B],axis=3), up4], axis=3)
    up4=conv_block(up4, 512, 32, 'deconv1', strides=(2, 2))
    up4=conv_block(up4, 512, 32, 'deconv2', strides=(2, 2))
    up4=conv_block(up4, 512, 32, 'deconv3', strides=(2, 2))
    
    # 64

    up3 = UpSampling2D((2, 2))(up4) 
    upc=concatenate([down2T,down2B], axis=3)
    up3 = concatenate([upc, up3], axis=3)
    up3=conv_block(up3, 512, 64, 'deconv1', strides=(2, 2))
    up3=conv_block(up3, 512, 64, 'deconv2', strides=(2, 2))
    up3=conv_block(up3, 512, 64, 'deconv3', strides=(2, 2))
    # 128

    up2 = UpSampling2D((2, 2))(up3) 
    upc = concatenate([down1T,down1B], axis=3)
    up2 = concatenate([upc, up2], axis=3)
    up2=conv_block(up2, 256, 128, 'deconv1', strides=(2, 2))
    up2=conv_block(up2, 256, 128, 'deconv2', strides=(2, 2))
    up2=conv_block(up2, 256, 128, 'deconv3', strides=(2, 2))
    # 256

    up1 = UpSampling2D((2, 2))(up2)  
    upc=concatenate([down0T,down0B], axis=3)
    up1 = concatenate([upc, up1], axis=3)
    up1=conv_block(up1, 128, 256, 'deconv1', strides=(2, 2))
    up1=conv_block(up1, 128, 256, 'deconv2', strides=(2, 2))
    up1=conv_block(up1, 128, 256, 'deconv3', strides=(2, 2))


    # 512

    up0a = UpSampling2D((2, 2))(up1)
    upc= concatenate([down0aT,down0aB],axis=3)
    up0a = concatenate([upc, up0a], axis=3)    
    up0a=conv_block(up0a, 64, 512, 'deconv1', strides=(2, 2))
    up0a=conv_block(up0a, 64, 512, 'deconv2', strides=(2, 2))
    up0a=conv_block(up0a, 64, 512, 'deconv3', strides=(2, 2))
  
    
    

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)
        

    model = Model(inputs=img_input, outputs=classify,name='YNet')
    optimizerc =CustomRMSprop(lr=0.00001,multipliers = LR_mult_dict)
    model.compile(optimizer=optimizerc, loss=bce_dice_loss, metrics=[dice_coeff])


    return model


def conv_block(input_tensor, filters, stage, block, strides=(2, 2)):
    conv_name_base = 'ColNet_' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_uniform')(input_tensor)
    x = BatchNormalization(name=bn_name_base)(x)
    x = Activation('selu')(x) 
    
    return x




def get_unet(input_shape=(512, 512, 3), num_classes=1):
    inputs = Input((img_rows, img_cols, img_ch))
    # 512

    down0a = Conv2D(16, (3, 3), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model
def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
    return loss
def dice_coef_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
def dice_coeff(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def optimize(num_iteration,historyfile='history.pickle'):
    save = {}
    print('-'*30)
    print('Trainig model...')
    print('-'*30)
    for i in range(num_iteration):
        print("Loading %d to %d" %(max(0, i)*32,(i+1)*32))
        x_batch, y_true_batch = random_batch(imgs_train, imgs_mask_train ,32)
        print('Data loaded...')
        history = model.fit(x_batch, y_true_batch, batch_size=5, epochs=3, verbose=1, shuffle=True,
              validation_split=0.2, callbacks=[model_checkpoint])
        save['cv'+ str(i)]= history.history
        del x_batch
        del y_true_batch
    f = open(historyfile, 'wb')
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
    f.close()



class CustomRMSprop(Optimizer):
    """RMSProp optimizer.
    It is recommended to leave the parameters of this optimizer
    at their default values
    (except the learning rate, which can be freely tuned).
    This optimizer is usually a good choice for recurrent
    neural networks.
    # Arguments
        lr: float >= 0. Learning rate.
        rho: float >= 0.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
        decay: float >= 0. Learning rate decay over each update.
    # References
        - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    """

    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0.,multipliers=None,
                 **kwargs):
        super(CustomRMSprop, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.rho = K.variable(rho, name='rho')
            self.decay = K.variable(decay, name='decay')
            self.iterations = K.variable(0, dtype='int64', name='iterations')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.lr_multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            if p.name in self.lr_multipliers:
                new_lr = lr * self.lr_multipliers[p.name]
            else:
                new_lr = lr                 
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))
            new_p = p - new_lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'rho': float(K.get_value(self.rho)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon}
        base_config = super(CustomRMSprop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
    
