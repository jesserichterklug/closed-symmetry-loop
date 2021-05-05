import tensorflow as tf
import numpy as np

epsilon = 0.00001 #tf.keras.backend.epsilon()
def norm(x):
    return tf.linalg.norm(x, axis=-1, keepdims = True)

def normalize(x):
    return x / (norm(x) + epsilon)

def cross(x, y):
    return tf.linalg.cross(x, y)

def cross_n(x,y):
    return normalize(cross(x,y))

def angle_between(x, y, dot_product='i, bxyi->bxy'):
    numerator = tf.einsum(dot_product, normalize(x), normalize(y))
    return tf.math.acos(tf.math.minimum(tf.math.maximum(numerator, epsilon - 1.), 1. - epsilon))

def get_Angle_around_Axis(axis, v_from, v_to, dot_product='bxyi, bxyi->bxy'):
    corrected_v_from = cross_n(cross(axis, v_from), axis)
    corrected_v_to = cross_n(cross(axis, v_to), axis)
    
    angle = angle_between(corrected_v_from, corrected_v_to, dot_product=dot_product)
    
    new_axis = cross_n(corrected_v_from, corrected_v_to)
    sign_correction_factor = tf.squeeze(tf.math.sign(tf.stop_gradient(norm(new_axis + axis) - 1.)), axis=-1)
    
    angle *= tf.math.minimum(sign_correction_factor * 2. + 1, 1)
    return angle


def change_Angle_around_Axis(axis, x, v_zero, factor, dot_product='bxyi, bxyi->bxy'):
    factor = factor if not np.isinf(factor) else 0
    
    current_angle = get_Angle_around_Axis(axis, v_zero, x, dot_product=dot_product)# + np.pi
    angle_change = current_angle * (factor - 1) 
    R_to_make_newX_from_X = make_R_from_angle_axis(angle_change, axis)
    return tf.squeeze(tf.matmul(R_to_make_newX_from_X, tf.expand_dims(x, axis=-1)), axis=-1)


def make_R_from_angle_axis(angle, axis):
    #https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)
    t = 1. - c
    x = axis[...,0]
    y = axis[...,1]
    z = axis[...,2]

    part_one = c[...,tf.newaxis,tf.newaxis] * tf.eye(3, dtype=c.dtype)

    part_two = t[...,tf.newaxis,tf.newaxis] * tf.stack([
        tf.stack([x*x, x*y, x*z], axis=-1),
        tf.stack([x*y, y*y, y*z], axis=-1),
        tf.stack([x*z, y*z, z*z], axis=-1)
    ], axis=-2)

    zero = tf.zeros_like(z)
    part_three = s[...,tf.newaxis,tf.newaxis] * tf.stack([
         tf.stack([zero, -z, y], axis=-1),
         tf.stack([z, zero, -x], axis=-1),
         tf.stack([-y, x, zero], axis=-1)
    ], axis=-2)

    return part_one + part_two + part_three   


def generate_px_coordinates(shape, coord_K, strides=1):
    u, v = tf.meshgrid(tf.range(shape[1], dtype=tf.float32), tf.range(shape[0], dtype=tf.float32))
    return u * coord_K[:,0:1,0:1] * strides + coord_K[:,1:2,0:1], v * coord_K[:,0:1,1:2] * strides + coord_K[:,1:2,1:2]