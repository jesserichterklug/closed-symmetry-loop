from tensorflow.keras.layers import Conv2D, Concatenate, UpSampling2D, Reshape, Lambda
from network.augmentations import ContrastNoise, ContrastNoiseSingle, GaussianNoise, BrightnessNoise
from tensorflow.image import rgb_to_grayscale

def dense_block(x, channels, iterations):
    for i in range(iterations):
        x1 = x
        x1 = Conv2D(channels, 1, padding='same',  activation ='selu')(x1)
        x1 =  Conv2D(channels, 3, padding='same',  activation ='selu')(x1)
        x = Concatenate()([x,x1])
    return Conv2D(channels * 2, 1, padding='same',  activation ='selu')(x)


def rgb255_to_obj_net(rgb):
    
    input_x_ = ContrastNoise(0.25)(rgb)
    input_x_ = ContrastNoiseSingle(0.25)(input_x_)
    input_x_ = GaussianNoise(.08 * 128)(input_x_)
    input_x_ = BrightnessNoise(0.2 * 128)(input_x_)
    
    rgb = Lambda(lambda x: rgb_to_grayscale(x) / 255.)(input_x_)

    
    x = Conv2D(8, 5, padding='same',   activation ='selu')(rgb)
    tier0 = x
    
    x = Conv2D(16,5,strides = 2,padding='same',   activation ='selu')(x)
    x = dense_block(x, 32, 3)
    tier1 = x
    
    x = Conv2D(64,5,strides = 2,padding='same',  activation ='selu')(x)
    x = dense_block(x, 64, 6)
    x = dense_block(x, 64, 6)
    tier2 = x
    
    x = Conv2D(128, 5, strides = 2, padding='same', activation ='selu')(x)
    x = dense_block(x, 64, 12)
    x = dense_block(x, 64, 12)
    tier3 = x
    
    x = Conv2D(128, 5, strides = 2, padding='same', activation ='selu')(x)
    x = dense_block(x, 128, 12)

    def up_path(x):
        x = UpSampling2D()(x)
        x = Concatenate()([x, tier3])
        x = Conv2D(64, 3, padding='same', activation ='selu')(x)
        x = dense_block(x, 32, 12)

        x = UpSampling2D()(x)
        x = Concatenate()([x, tier2])
        x = Conv2D(32, 3, padding='same', activation ='selu')(x)
        x = dense_block(x, 16, 6)

        x = UpSampling2D()(x)
        x = Concatenate()([x, tier1])
        x = Conv2D(24, 3, padding='same', activation ='selu')(x)
        x = dense_block(x, 12, 4)

        return x
    
    star_x = up_path(x)
    star = Conv2D(3, 1, padding='same',  activation=None, name='star')(star_x)
    
    dash_x = up_path(x)
    dash = Conv2D(3, 1, padding='same',  activation=None, name='dash')(dash_x)
    
    w_px_x = up_path(x)
    w_px = Conv2D(4, 1, padding='same',  activation=None, name='wpx')(w_px_x)
    
    w_d_x = up_path(x)
    w_d = Conv2D(1, 1, padding='same',  activation=None, name='wd')(w_d_x)
    
    seg_x = up_path(x)
    seg = Conv2D(1, 1, padding='same',  activation='sigmoid')(seg_x)
        
    return star, dash, w_px, w_d, seg