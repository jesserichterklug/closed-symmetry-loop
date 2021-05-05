import tensorflow as tf
from tensorflow.keras.layers import Lambda, Multiply

from utils import generate_px_coordinates

def create_Dataset_conversion_layers(xDim, yDim, model_info, strides=1):
    inputs = {
        'rgb':             tf.keras.Input(shape=(yDim,xDim,3,)),
        'depth':           tf.keras.Input(shape=(yDim,xDim,)),
        'segmentation':    tf.keras.Input(shape=(yDim,xDim,), dtype=tf.int32),
        'camera_matrix':   tf.keras.Input(shape=(3,3,)),
        'coord_offset':    tf.keras.Input(shape=(2,2,)),
        'roationmatrix':   tf.keras.Input(shape=(3,3,)),
        'translation':     tf.keras.Input(shape=(3,)),
    }
    
    depth = Lambda(lambda x: x[:,::strides,::strides])(inputs['depth'])
    segmentations = Lambda(lambda x: x[:,::strides,::strides])(inputs['segmentation'])
    segmentations = Lambda(lambda x: tf.cast(x[...,tf.newaxis] > 0, tf.float32))(segmentations)
    
    def depth_based_cam_coords(var):
        depth, cam_K, coord_K = var
#         u, v = tf.meshgrid(tf.range(depth.shape[2], dtype=tf.float32), tf.range(depth.shape[1], dtype=tf.float32))
#         u, v = (u * coord_K[:,0:1,0:1] * strides + coord_K[:,1:2,0:1], v * coord_K[:,0:1,1:2] * strides + coord_K[:,1:2,1:2])
        u, v = generate_px_coordinates(depth.shape[1:3], coord_K, strides)
        scaled_coords = tf.stack([u * depth, v * depth, depth], axis=-1)
        return tf.einsum('bij,bxyj->bxyi', tf.linalg.inv(cam_K), scaled_coords)
    cam_coords = Lambda(depth_based_cam_coords)((depth, inputs['camera_matrix'], inputs['coord_offset']))
    
    def cam_to_obj(var):
        R, t, cam_coords = var
        return tf.einsum('bji,byxj->byxi', R, cam_coords - t[:, tf.newaxis, tf.newaxis])
    obj_image = Lambda(cam_to_obj)((inputs['roationmatrix'], inputs['translation'], cam_coords))
#     obj_image = Multiply()([obj_image, segmentations])
    
    def obj_validity(obj_image):
        obj_mins = tf.constant(model_info['mins'], tf.float32) * 1.1
        obj_maxs = tf.constant(model_info['maxs'], tf.float32) * 1.1
                
        obj_dim_in_range = tf.math.logical_and(tf.math.less(obj_mins, obj_image), tf.math.less(obj_image, obj_maxs))
        obj_dim_in_range = tf.cast(tf.math.reduce_all(obj_dim_in_range, axis=-1, keepdims=True), tf.float32)
        return obj_dim_in_range
    
    isvalid = Lambda(obj_validity)(obj_image)
    isvalid = Multiply()([isvalid, segmentations])
    obj_image = Multiply()([obj_image, isvalid])
    
    
#     depth = Lambda(lambda x: x[:,::strides, ::strides])(inputs['depth'])
    segmentation = Lambda(lambda x: x[:,::strides, ::strides])(inputs['segmentation'])
    
    return inputs, obj_image, isvalid, depth, segmentation