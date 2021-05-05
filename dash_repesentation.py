import tensorflow as tf

import sys
sys.path.insert(0,'..')
from utils import *


'''
Dash-representation (without camera distortion) from object rotation and object points  
'''
class DashRepresentation(tf.keras.layers.Layer):
    def __init__(self, offset, **kwargs):
        super(DashRepresentation, self).__init__(**kwargs)
        self.supports_masking = True
        
        self.offset = offset
        
    def call(self, R, po):
        return tf.einsum('bij,byxj->byxi', R, po) + self.offset
        
    def get_config(self):
        config = {'offset': self.offset}
        base_config = super(DashRepresentation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    


class RemoveCameraEffect(tf.keras.layers.Layer):
    def __init__(self, strides=1, **kwargs):
        super(RemoveCameraEffect, self).__init__(**kwargs)
        self.supports_masking = True
        self.strides = strides
        
    def get_config(self):
        config = {'strides' : self.strides }
        base_config = super(RemoveCameraEffect, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, v_cam, cam_K, coord_K):
        return tf.einsum('bxyij, bxyj-> bxyi', self.make_Rpxy(v_cam.shape[1:3], cam_K, coord_K), v_cam)
    
#     def make_R_from_angle_axis(self, angle, axis):
#         #https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToMatrix/
#         c = tf.math.cos(angle)
#         s = tf.math.sin(angle)
#         t = 1. - c
#         x = axis[...,0]
#         y = axis[...,1]
#         z = axis[...,2]

#         part_one = c[...,tf.newaxis,tf.newaxis] * tf.eye(3)

#         part_two = t[...,tf.newaxis,tf.newaxis] * tf.stack([
#             tf.stack([x*x, x*y, x*z], axis=-1),
#             tf.stack([x*y, y*y, y*z], axis=-1),
#             tf.stack([x*z, y*z, z*z], axis=-1)
#         ], axis=-2)

#         zero = tf.zeros_like(z)
#         part_three = s[...,tf.newaxis,tf.newaxis] * tf.stack([
#              tf.stack([zero, -z, y], axis=-1),
#              tf.stack([z, zero, -x], axis=-1),
#              tf.stack([-y, x, zero], axis=-1)
#         ], axis=-2)

#         return part_one + part_two + part_three
    
    def make_Rpxy(self, shape, cam_K, coord_K):
        f = tf.stack([cam_K[:,0,0], cam_K[:,1,1]], axis=-1)
        c = cam_K[:,:2,2]

        u, v = generate_px_coordinates(shape, coord_K, self.strides)
        coords_c = tf.stack([u - c[:,0][:,tf.newaxis,tf.newaxis],
                             v - c[:,1][:,tf.newaxis,tf.newaxis]
                            ], axis=-1)

        coords_3d_with_z1 = tf.concat([coords_c / f[:,tf.newaxis,tf.newaxis], tf.ones_like(coords_c[:,:,:,:1])], axis=-1)
        z = tf.constant([0,0,1], dtype=coords_3d_with_z1.dtype)

        axes = tf.linalg.cross(z * tf.ones_like(coords_3d_with_z1), coords_3d_with_z1)
        axes /= tf.norm(axes, axis=-1, keepdims=True) + 0.000001

        coords_3d_with_z1 = tf.debugging.assert_all_finite(coords_3d_with_z1, 'coords_3d_with_z1 is not finite', name=None)
        angles = angle_between(z, coords_3d_with_z1)

        angles = tf.debugging.assert_all_finite(angles, 'angles is not finite', name=None)
        axes = tf.debugging.assert_all_finite(axes, 'axes is not finite', name=None)
        RpxRpy = make_R_from_angle_axis(angles, axes)   
#         print('RpxRpy', RpxRpy.shape)
        RpxRpy = tf.debugging.assert_all_finite(RpxRpy, 'RpxRpy is not finite', name=None)
        
        return RpxRpy
