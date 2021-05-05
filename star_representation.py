from utils import *
import numpy as np
from math import isclose    

def collapses_obj_to_dot_symmetry(obj, x_factor=1, y_factor=1, z_factor=1):
    R = tf.eye(3, batch_shape=tf.shape(obj)[:-1])
    
    obj = change_Angle_around_Axis(R[...,0], obj, R[...,1], x_factor)
    obj = change_Angle_around_Axis(R[...,1], obj, R[...,2], y_factor)
    obj = change_Angle_around_Axis(R[...,2], obj, R[...,0], z_factor)
    
    return obj

class StarRepresentation(tf.keras.layers.Layer):
    def __init__(self, model_info, **kwargs):
        super(StarRepresentation, self).__init__(**kwargs)
        self.supports_masking = True
        
        self.model_info = model_info
        
    def call(self, po):
        if self.model_info["symmetries_continuous"]:
            print("Starring as symmetries_continuous")
            return collapses_obj_to_dot_symmetry(po, z_factor=np.inf)

        if len(self.model_info["symmetries_discrete"]) == 0:
            print("Starring is not changing anything")
            return po

        if isclose(self.model_info["symmetries_discrete"][0][2,2], 1, abs_tol=1e-3):
            offset = self.model_info["symmetries_discrete"][0][:3,-1] / 2.
            po = po + offset
            print("po was corrected by", offset)

            print("Starring as symmetries_discrete with z_factor=", len(self.model_info["symmetries_discrete"])+1)
            return collapses_obj_to_dot_symmetry(po, z_factor=len(self.model_info["symmetries_discrete"])+1)


        if isclose(self.model_info["symmetries_discrete"][0][1,1], 1, abs_tol=1e-3):
            offset = self.model_info["symmetries_discrete"][0][:3,-1] / 2.
            po = po + offset
            print("po was corrected by", offset)

            print("Starring as symmetries_discrete with y_factor=", len(self.model_info["symmetries_discrete"])+1)
            return collapses_obj_to_dot_symmetry(po, y_factor=len(self.model_info["symmetries_discrete"])+1)

        assert(False)
        
    def get_config(self):
        config = {'model_info': self.model_info}
        base_config = super(StarRepresentation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    