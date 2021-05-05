import tensorflow as tf
import numpy as np
from math import isclose    

import sys
sys.path.insert(0,'..')
from utils import *

def get_obj_star0_from_obj_star(obj_star,
                                x_factor=1, y_factor=1, z_factor=1):
    R = tf.eye(3, batch_shape=tf.shape(obj_star)[:-1])
    
    obj_star = change_Angle_around_Axis(R[...,2], obj_star, R[...,0], 1. / z_factor)
    obj_star = change_Angle_around_Axis(R[...,1], obj_star, R[...,2], 1. / y_factor)
    obj_star = change_Angle_around_Axis(R[...,0], obj_star, R[...,1], 1. / x_factor)
    
    return obj_star


class PODestarisation(tf.keras.layers.Layer):
    def __init__(self, 
                 model_info,
                 amount_of_instances = 6,
                 **kwargs):
        super(PODestarisation, self).__init__(**kwargs)
        self.supports_masking = True
        
        self.model_info = model_info
        self.amount_of_instances = amount_of_instances
        
    def get_config(self):
        config = {
            'amount_of_instances': self.amount_of_instances,
            'model_info': self.model_info
        }
        base_config = super(PODestarisation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def call(self, postar, vo, iseg, train_R=None):
#         print(train_R)
        def angle_substraction(a,b):
            return tf.minimum(tf.abs(a-b), tf.abs(tf.minimum(a,b) + np.pi * 2. - tf.maximum(a,b)))

        def best_symmetrical_po(star0, factor, axis):
            if train_R is None:
                ref = self.generate_ref_data([postar, vo, iseg], 15,  3)
            else:
                gt_ref_data = {
                    'po': tf.eye(3, batch_shape=(tf.shape(train_R)[:1]))[:,tf.newaxis],
                    'vo': tf.transpose(train_R[:,tf.newaxis], perm=[0,1,3,2])
                }
                ref = gt_ref_data
#             print(ref)
            dash_angles = angle_between(vo, ref['vo'], 'byxi, boji->byxoj')
#             print('dash_angles', dash_angles, vo.shape, ref['vo'].shape)


            symR = make_R_from_angle_axis(2*np.pi / factor, axis)

            allp_pos = star0[...,tf.newaxis,:]
            for _ in range(factor-1):
                newp = tf.einsum('ij, byxj -> byxi', symR, allp_pos[...,-1,:])
                allp_pos = tf.concat([allp_pos, newp[...,tf.newaxis,:]], axis=-2)


            allp_po_angles = angle_between(tf.stop_gradient(allp_pos), ref['po'], 'byxsi, boji->byxosj')
            allp_angle_diffs = tf.reduce_sum(angle_substraction(allp_po_angles, tf.expand_dims(dash_angles, axis=-2))**2, axis=-1)

#             print('allp_angle_diffs','allp_angle_diffs', allp_angle_diffs.shape)
            arg_min = tf.math.argmin(allp_angle_diffs, axis=-1)
            best_po = tf.gather_nd(allp_pos, tf.expand_dims(arg_min, axis=-1), batch_dims=3) 
#             print('allp_angle_diffs', allp_pos.shape, best_po.shape)

            o_wide_error = tf.reduce_sum(tf.reduce_min(allp_angle_diffs, axis=-1)[...,tf.newaxis,:] * iseg[...,tf.newaxis], axis=[1,2], keepdims=True)

    #         tf.print(o_wide_error, o_wide_error.shape, 'o_wide_error')
            arg_min = tf.math.argmin(o_wide_error, axis=-1)
    #         tf.print(tf.gather_nd(o_wide_error, tf.expand_dims(arg_min, axis=-1), batch_dims=4) , tf.gather_nd(o_wide_error, tf.expand_dims(arg_min, axis=-1), batch_dims=4).shape)
    #         tf.print(arg_min, arg_min.shape)
            arg_min_ = arg_min * tf.cast(iseg, arg_min.dtype)
#             print('allp_angle_diffs_best_po_befoire', best_po.shape)
            best_po = tf.gather_nd(best_po, tf.expand_dims(arg_min_, axis=-1), batch_dims=3) 
#             print('allp_angle_diffs_arg_min', arg_min.shape, arg_min_.shape)
#             print('allp_angle_diffs_best_po', best_po.shape, tf.reduce_sum(best_po, axis=-2).shape)

            return tf.reduce_sum(best_po * iseg[..., tf.newaxis], axis=-2)

        def best_continues_po(star0, axis):
            
            if train_R is None:                
                def direct_calc_z_dir(x):
                    postar, vo_image, seg = x
                    def make_equation_matrix_pd(vo, postar):
                        A = vo[...,tf.newaxis,:]
                        AtA = tf.matmul(A,A, transpose_a=True)
                        b = postar[...,2:,tf.newaxis]
                        Atb = tf.matmul(A,b, transpose_a=True)
                        return AtA, Atb

                    AtA, Atb = make_equation_matrix_pd(vo_image * seg, postar*seg)

                    _AtA = tf.reduce_sum(AtA,axis=[1,2])
                    _Atb = tf.reduce_sum(Atb,axis=[1,2])

                    return tf.matmul(tf.linalg.pinv(_AtA), _Atb)#[...,0]    


                def make_oracle_R(Rz):
                    z = normalize(Rz[...,0])
                    o = tf.concat([z[...,1:], z[...,:1] ],axis=-1)
                    x = cross(z,o)
                    y = cross(z,x)
                    z = cross(x,y)
                    return tf.stack([x,y,z], axis=-1)
                
                _R = make_oracle_R(direct_calc_z_dir([postar, vo, iseg]))
                
            else:
                _R = train_R
                
            ref = {
                'po': tf.eye(3, batch_shape=(tf.shape(_R)[:1]))[:,tf.newaxis],
                'vo': tf.transpose(_R[:,tf.newaxis], perm=[0,1,3,2])
            }
#             print(ref)
            dash_angles = angle_between(vo, ref['vo'], 'byxi, boji->byxoj')
#             print('dash_angles', dash_angles, vo.shape, ref['vo'].shape)


    #         star0 = star0[...,tf.newaxis,:] * iseg[...,tf.newaxis]
#             print('star0_under_ref', star0.shape, tf.ones_like(ref['po']).shape)
            star0_ = star0[...,tf.newaxis,tf.newaxis,:] * tf.ones_like(ref['po'])[:,tf.newaxis,tf.newaxis]
            star0_under_ref = change_Angle_around_Axis(
                axis * tf.ones_like(star0_), # axis
                star0_, # x
                ref['po'][:,tf.newaxis,tf.newaxis] * tf.ones_like(star0_), # v_zero
                0, # factor
                'byxoji ,byxoji ->byxoj '#'bxysoi,bxysoi->bxyso'
            )
#             print(star0_under_ref, 'star0_under_ref')
            po_star0_angles = angle_between(star0_under_ref, ref['po'],  'byxoji, boji->byxoj')

#             print(po_star0_angles, 'star0_under_ref')
            beta_upper_part = tf.math.cos(dash_angles)
            beta_lower_part = tf.math.cos(po_star0_angles)

            #this uses reduce for min/max, therefore a nan (from division) will be exchange by the comparison value
            quotient = beta_upper_part / beta_lower_part
            quotient = tf.reduce_min(tf.stack([quotient, tf.ones_like(quotient) - 0.0001], axis=-1), axis=-1)
            quotient = tf.reduce_max(tf.stack([quotient, -tf.ones_like(quotient) + 0.0001], axis=-1), axis=-1)
            beta = tf.math.acos(quotient)

            R_betas = make_R_from_angle_axis(tf.stack([beta,-beta],axis=-1), axis)

            allp_pos = tf.einsum('byxojaki,byxoji->byxojak', R_betas, star0_under_ref)
            allp_pos = tf.concat([allp_pos[...,0,:], allp_pos[...,1,:]], axis=-2)

            allp_po_angles = angle_between(tf.stop_gradient(allp_pos), ref['po'], 'byxosi, boji->byxosj')
            allp_angle_diffs = tf.reduce_sum(angle_substraction(allp_po_angles, tf.expand_dims(dash_angles, axis=-2))**2, axis=-1)

            arg_min = tf.math.argmin(allp_angle_diffs, axis=-1)
            best_po = tf.gather_nd(allp_pos, tf.expand_dims(arg_min, axis=-1), batch_dims=4) 
#             print('allp_angle_diffs', allp_pos.shape, best_po.shape)

            o_wide_error = tf.reduce_sum(tf.reduce_min(allp_angle_diffs, axis=-1)[...,tf.newaxis,:] * iseg[...,tf.newaxis], axis=[1,2], keepdims=True)
            arg_min = tf.math.argmin(o_wide_error, axis=-1)
            arg_min_ = arg_min * tf.cast(iseg, arg_min.dtype)
    #         arg_min = tf.math.argmin(tf.reduce_min(allp_angle_diffs, axis=-1)[...,tf.newaxis,:] * iseg[...,tf.newaxis], axis=-1)
            best_po = tf.gather_nd(best_po, tf.expand_dims(arg_min_, axis=-1), batch_dims=3)
#             print('allp_angle_diffs_fin', arg_min.shape, best_po.shape, tf.reduce_sum(best_po, axis=-2).shape)

            return tf.reduce_sum(best_po * iseg[..., tf.newaxis], axis=-2)

        if self.model_info["symmetries_continuous"]:
            print("destarring as symmetries_continuous")
            return best_continues_po(postar, tf.constant([0,0,1],tf.float32))

        if len(self.model_info["symmetries_discrete"]) == 0:
            print("destarring is not changing anything")
            return postar

        if isclose(self.model_info["symmetries_discrete"][0][2,2], 1, abs_tol=1e-3):
            factor = len(self.model_info["symmetries_discrete"])+1
            print("destarring as symmetries_discrete with z_factor=", factor)
            po_ = best_symmetrical_po(get_obj_star0_from_obj_star(postar, z_factor=factor), factor, tf.constant([0,0,1],tf.float32))

            offset = self.model_info["symmetries_discrete"][0][:3,-1] / 2.
            print("po_ was corrected by", -offset)
            return po_ - offset

        if isclose(self.model_info["symmetries_discrete"][0][1,1], 1, abs_tol=1e-3):
            factor = len(self.model_info["symmetries_discrete"])+1
            print("destarring as symmetries_discrete with y_factor=", factor)
            po_ = best_symmetrical_po(get_obj_star0_from_obj_star(postar, y_factor=factor), factor, tf.constant([0,1,0],tf.float32))

            offset = self.model_info["symmetries_discrete"][0][:3,-1] / 2.
            print("po_ was corrected by", -offset)
            return po_ - offset
        
        assert(False)

    @tf.function
    def generate_ref_data(self, x, counts, sample_size):
        postar, vo, iseg = x

        def generate_samples_per_batch(sb_x):
            sb_postar, sb_vo, sb_iseg = sb_x

            def generate_samples_per_instance(i):
                si_seg = sb_iseg[:,:,i]
                selection_index = tf.where(si_seg > 0.5)
                if tf.shape(selection_index)[0] > 0:
                    vo_sel = tf.gather_nd(sb_vo, selection_index)
                    postar_sel = tf.gather_nd(sb_postar, selection_index)

                    pos = tf.random.uniform((counts, sample_size, 1),
                                            minval=0, maxval=tf.shape(vo_sel)[0], 
                                            dtype=tf.dtypes.int32, seed=55)

                    vo_samples = tf.gather_nd(vo_sel, pos)
                    postar_samples = tf.gather_nd(postar_sel, pos)
                else:
                    vo_samples = tf.ones((counts, sample_size, 3))     # a.k.a zeros but this would interprete to zero angle to any other vector
                    postar_samples = tf.ones((counts, sample_size, 3)) # a.k.a zeros but this would interprete to zero angle to any other vector

                return tf.stack([vo_samples, postar_samples], axis=-1)
            return tf.concat([generate_samples_per_instance(i) for i in range(self.amount_of_instances)], axis=0)


        samples = tf.map_fn(generate_samples_per_batch,
                                   [postar, vo, iseg], 
                                   fn_output_signature=tf.float32
                                  ) 

        vo_samples = samples[...,0]
        postar_samples = samples[...,1]

        def angle_substraction(a,b):
            return tf.minimum(tf.abs(a-b), tf.abs(tf.minimum(a,b) + np.pi * 2. - tf.maximum(a,b)))

        def make_ref_outof_samples_symmetrical(star0_samples, factor, axis):
            
#             print('refoutof...  coninues', vo_samples, postar_samples, f'assumed (b, {self.amount_of_instances * counts}, 3, 3)')
#             star0_samples = get_obj_star0_from_obj_star(postar_samples, z_factor=factor)

            ref_vo = vo_samples
            dash_angles = angle_between(vo_samples, ref_vo, dot_product='bski, bsji->bsjk')

            symR = make_R_from_angle_axis(2*np.pi / factor, axis)

            allp_pos = star0_samples[...,tf.newaxis,:]
            for _ in range(factor-1):
                newp = tf.einsum('ij, bskj -> bski', symR, allp_pos[...,-1,:])
                allp_pos = tf.concat([allp_pos, newp[...,tf.newaxis,:]], axis=-2)

#             print(allp_pos.shape)
            assert(sample_size == 3)
            mg = np.meshgrid(np.arange(factor), np.arange(factor), np.arange(factor))
            gather_per = tf.constant(np.stack(mg, axis=0).reshape((sample_size,-1)))
            gather_per = gather_per[...,tf.newaxis] *  tf.ones_like(allp_pos[:,:,:,:1,:1], gather_per.dtype)
            gather_per_rev = tf.constant(np.stack(mg, axis=-1).reshape((-1,sample_size)))
#             print(gather_per)
            all_combi = tf.gather_nd(allp_pos, gather_per, batch_dims=3) 
            print(all_combi)

            all_combi_po_angles = angle_between(all_combi, all_combi, dot_product='bskni, bsjni->bsjkn')
#             print('all_combi_po_angles',all_combi_po_angles.shape)

            all_combi_angle_diffs = tf.reduce_sum(angle_substraction(all_combi_po_angles, tf.expand_dims(dash_angles, axis=-1))**2, axis=[-2,-3])
#             print('all_combi_angle_diffs',all_combi_angle_diffs.shape)

            arg_min = tf.math.argmin(all_combi_angle_diffs, axis=-1)
#             print('arg_min',arg_min.shape, gather_per_rev.shape)

            arg_min_combi = tf.gather_nd(gather_per_rev, tf.expand_dims(arg_min, axis=-1), batch_dims=0)
#             print('arg_min_combi',arg_min_combi.shape, allp_pos.shape)

            best_pos = tf.gather_nd(allp_pos, tf.expand_dims(arg_min_combi, axis=-1), batch_dims=3) 
#             print('best_pos',best_pos.shape)

            ref_po = best_pos

            return (ref_vo, ref_po)
    
        if self.model_info["symmetries_continuous"]:
            print("generate ref samples for continuous symmetries around z")
            
            ref_vo, ref_po = make_ref_outof_samples_continues(tf.constant([0,0,1],tf.float32))

        elif isclose(self.model_info["symmetries_discrete"][0][2,2], 1, abs_tol=1e-3):
            factor = len(self.model_info["symmetries_discrete"])+1
            print("generate ref samples discrete symmetries with z_factor=", factor)
            
            ref_vo, ref_po = make_ref_outof_samples_symmetrical(get_obj_star0_from_obj_star(postar_samples, z_factor=factor),factor, tf.constant([0,0,1],tf.float32))

        elif isclose(self.model_info["symmetries_discrete"][0][1,1], 1, abs_tol=1e-3):
            factor = len(self.model_info["symmetries_discrete"])+1
            print("generate ref samples discrete symmetries with y_factor=", factor)
            
            ref_vo, ref_po = make_ref_outof_samples_symmetrical(get_obj_star0_from_obj_star(postar_samples, y_factor=factor), factor, tf.constant([0,1,0],tf.float32))
        else:
            assert(False)

        return { 'po': ref_po, 'vo': ref_vo }