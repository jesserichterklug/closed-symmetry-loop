import tensorflow as tf
import numpy as np

from utils import epsilon, generate_px_coordinates

class AvgSqrDiff_of_validPixels(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AvgSqrDiff_of_validPixels, self).__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, image0, image1, isvalid):
        error = (image0 - image1)**2 * isvalid
        return tf.reduce_sum(error, axis=[1,2,3]) / (epsilon + tf.reduce_sum(isvalid, axis=[1,2,3]))

class Po_to_Img(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Po_to_Img, self).__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, po, cam_K, R, t):
        in_cam = tf.einsum('bij,byxj->byxi', R, po) + t[:, tf.newaxis, tf.newaxis]
        in_img = tf.einsum('bij,bxyj->bxyi', cam_K, in_cam)
        
        return in_img[...,:2] / (in_img[...,2:] + epsilon), in_cam
    
class UV_diff(tf.keras.layers.Layer):
    def __init__(self, strides, **kwargs):
        super(UV_diff, self).__init__(**kwargs)
        self.supports_masking = True
        self.strides = strides
        
    def get_config(self):
        config = {'strides' : self.strides }
        base_config = super(UV_diff, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, x, coord_K):
#         u, v = tf.meshgrid(tf.range(x.shape[2], dtype=tf.float32), tf.range(x.shape[1], dtype=tf.float32))
#         u, v = (u * coord_K[:,0:1,0:1] + coord_K[:,1:2,0:1], v * coord_K[:,0:1,1:2] + coord_K[:,1:2,1:2])
        u, v = generate_px_coordinates(x.shape[1:3], coord_K, self.strides)
        return x - tf.stack([u, v], axis=-1)
    
class D_diff(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(D_diff, self).__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, po, depth):
        return po[...,2:] - depth[...,tf.newaxis]

class ToOmega(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ToOmega, self).__init__(**kwargs)
        self.supports_masking = True
        
    def call(self,  w, isvalid=None):
        assert(w.shape[-1] in [1,4])
        
        if not isvalid is None:
            w = w * isvalid
        
        if w.shape[-1] == 1:
            return w**2
        
        if w.shape[-1] == 4:
            A1 = w[...,0]
            A2 = w[...,1]
            A3 = w[...,2]
            A4 = w[...,3]
            result_shape = [tf.shape(w)[0], w.shape[1], w.shape[2], 2, 2]
            return tf.reshape(tf.stack([A1*A1+A3*A3, A2*A1+A4*A3, A1*A2+A3*A4, A2*A2+A4*A4], axis=-1), result_shape)

class Avg_nllh(tf.keras.layers.Layer):
    def __init__(self, pixel_cap = 100, **kwargs):
        super(Avg_nllh, self).__init__(**kwargs)
        self.supports_masking = True
        
        self.pixel_cap = pixel_cap
        
    def call(self, Omega, diff, isvalid):
        assert(Omega.shape[-1] == diff.shape[-1])
        assert(Omega.shape[-1] in [1,2])
        
        left_part = self.left_part(Omega)
        chi2_part = self.chi2_part(Omega, diff)
        
        error = tf.math.minimum(left_part + 0.5 * chi2_part, self.pixel_cap) * isvalid[...,0]
        chi2error = chi2_part * isvalid[...,0]
        
        avg_error = tf.reduce_sum(error, axis=[1,2])
        avg_chi2error = tf.reduce_sum(chi2error, axis=[1,2])
        divisor = tf.reduce_sum(isvalid[...,0], axis=[1,2]) + epsilon
        return avg_error / divisor, avg_chi2error / divisor
    
    def left_part(self, Omega):
        const_part = tf.constant(np.log((2*np.pi)**(Omega.shape[-1])),tf.float32)

        if Omega.shape[-1] == 1:
            var_part_pre_log = tf.squeeze(Omega**2, axis=-1)
        if Omega.shape[-1] == 2:
            Omega = tf.debugging.assert_all_finite(Omega, 'leftside_px omega entry is not finit')
            Omega += tf.eye(tf.shape(Omega)[-1]) * epsilon * (1. + tf.reduce_max(Omega,axis=[-2,-1],keepdims=True))
            var_part_pre_log = Omega[...,0,0] * Omega[...,1,1] - Omega[...,1,0] * Omega[...,0,1]#tf.linalg.det(Omega, name="the_Det")

        var_part = tf.math.log(var_part_pre_log + epsilon)
        
        return 0.5 * (const_part - var_part)
    
    def chi2_part(self, Omega, diff):
        
        if Omega.shape[-1] == 1:
            return tf.squeeze(diff**2 * Omega**2, axis=-1)
        
        if Omega.shape[-1] == 2:
            b1 = diff[...,0]
            b2 = diff[...,1]
            A1 = Omega[...,0,0]
            A2 = Omega[...,0,1]
            A3 = Omega[...,1,0]
            A4 = Omega[...,1,1]
            return b1**2 * A1 + b1 * b2 * (A2 + A3) + b2**2 * A4
            #return tf.squeeze(tf.matmul(diff, tf.matmul(Omega, diff),transpose_a=True), axis=[-2,-1])
        
    def get_config(self):
        config = {'pixel_cap': self.pixel_cap}
        base_config = super(Avg_nllh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    

class Seg_Loss(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Seg_Loss, self).__init__(**kwargs)
        self.supports_masking = True
        
    def call(self, sigmoid, labels):
        labels = tf.cast(labels[..., tf.newaxis], tf.float32) / 255.
        return (labels * -tf.math.log(sigmoid + epsilon) + (1. - labels) * -tf.math.log(1 - sigmoid + epsilon), #loss
                1. - tf.reduce_mean(tf.abs(labels - tf.cast(sigmoid > 0.5, sigmoid.dtype))), #percent
                1. - tf.reduce_sum(labels * tf.abs(labels -  tf.cast(sigmoid > 0.5, sigmoid.dtype))) / (epsilon + tf.reduce_sum(labels)) #fg_percent
               )

