import json, lycon, os
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm

def open_annotator(name):
    assert(Path(name).is_file())
    
    with open(name) as f:
        return json.load(f)

# build a list with data for all occurencies of that object 
def load_gt_data(root_dirs, oiu):
    found_data = []
    
    for rd in root_dirs:
        for root, sub_dirs, files in os.walk(rd):
            for sd in tqdm(sub_dirs):
                dir = f'{root}/{sd}'
                
                scene_gt = open_annotator(f'{dir}/scene_gt.json')
                scene_gt_info = open_annotator(f'{dir}/scene_gt_info.json')
                scene_camera = open_annotator(f'{dir}/scene_camera.json')
                
                assert(len(scene_gt) == len(scene_gt_info))
                assert(len(scene_gt) == len(scene_camera))
                
                for key, gt_values in scene_gt.items():
                    for vi, v in enumerate(gt_values):
                        if v["obj_id"] == oiu and scene_gt_info[key][vi]["visib_fract"] > 0.1: 
                            
                            new_data = {}
                            new_data['root'] = dir
                            new_data['file_name'] = "{:06d}".format(int(key))
                            new_data['oi_name'] = "{:06d}".format(vi)
                            new_data['cam_R_m2c'] = np.array(v["cam_R_m2c"]).reshape((3,3))
                            new_data['cam_t_m2c'] = np.array(v["cam_t_m2c"])
                            
                            bbox_obj = scene_gt_info[key][vi]["bbox_obj"]
                            new_data['bbox_start'] =np.array(bbox_obj[:2])
                            new_data['bbox_dims'] = np.array(bbox_obj[2:])
                            
                            new_data['cam_K'] = np.array(scene_camera[key]["cam_K"]).reshape((3,3))
                            new_data['depth_scale'] = scene_camera[key]["depth_scale"]
                            
                            new_data['visib_fract'] = scene_gt_info[key][vi]["visib_fract"]
                            
                            found_data.append(new_data)
            break
    return found_data

def load_foreign_data(root_dirs, foreign_info, oiu):
    found_data = []
    
    for rd in root_dirs:
        for root, sub_dirs, files in os.walk(rd):
            for sd in tqdm(sub_dirs):
                dir = f'{root}/{sd}'
                
                scene_foreign_info = open_annotator(f'{dir}/{foreign_info}')
                scene_camera = open_annotator(f'{dir}/scene_camera.json')
                
                assert(len(scene_foreign_info) == len(scene_camera))
                
                for key, foreign_values in scene_foreign_info.items():
                    for vi, v in enumerate(foreign_values):
                        if v["obj_id"] == oiu and v["score"] > 0.0:
                            
                            new_data = {}
                            new_data['root'] = dir
                            new_data['file_name'] = "{:06d}".format(int(key))
                            new_data['oi_name'] = "{:06d}".format(vi)
                                                        
                            bbox_obj = np.array(v["bbox_obj"]).astype(float)
                            new_data['bbox_start'] =bbox_obj[:2]
                            new_data['bbox_dims'] = bbox_obj[2:] - bbox_obj[:2]
#                             new_data['bbox_dims'] = np.array(bbox_obj[2:])
                            
                            new_data['cam_K'] = np.array(scene_camera[key]["cam_K"]).reshape((3,3))
                            new_data['depth_scale'] = scene_camera[key]["depth_scale"]
                            
                            new_data['score'] = v["score"]
                                                        
                            found_data.append(new_data)
            break
            
    return found_data


def load_data_item(datum, test_mode=False):
    img = lycon.load(f'{datum["root"]}/rgb/{datum["file_name"]}{".png" if "primesense" in datum["root"] else ".jpg"}')
    depthimg = np.array(Image.open(f'{datum["root"]}/depth/{datum["file_name"]}.png'), np.float32)
    depthimg *= datum["depth_scale"]

    if test_mode:
        return img, depthimg, datum["cam_K"], datum['bbox_start'], datum['bbox_dims']
        
    seg = lycon.load(f'{datum["root"]}/mask_visib/{datum["file_name"]}_{datum["oi_name"]}.png')[:,:,0]
    return img, depthimg, seg, datum["cam_K"], datum["cam_R_m2c"], datum["cam_t_m2c"], datum['bbox_start'], datum['bbox_dims']

def extract_item(datum, xyDim, sigma=0.2, test_mode=False):
    if test_mode:
        img, depth, cam_K, bbs, bbd = datum
        
        scale = bbd.max() / xyDim
        new_bbs = bbs + (bbd - bbd.max()) / 2
    else:
        img, depth, seg, cam_K, R, t, bbs, bbd = datum
        
        scale_diff = np.maximum(np.random.normal(1, sigma), 0.6)
        scale = bbd.max() / xyDim * scale_diff
        new_bbs = bbs + (bbd - bbd.max()) / 2 - (scale_diff- 1) * bbd.max() / 2  + np.random.normal(0, sigma * bbd.max() / 2., 2)

    transformation = [scale, 0, new_bbs[0], 0.0, scale,  new_bbs[1], 0.0, 0.0]
    coord_K = np.stack([np.array([scale,scale]), new_bbs])
    
    transformed_img = tfa.image.transform(img, transformation, interpolation='bilinear', output_shape=(xyDim,xyDim))
    transformed_depth = tfa.image.transform(depth, transformation, interpolation='bilinear', output_shape=(xyDim,xyDim))
    
    if test_mode:
        return transformed_img, transformed_depth, cam_K, coord_K
    else:
        transformed_seg = tfa.image.transform(seg, transformation, interpolation='bilinear', output_shape=(xyDim,xyDim))
        return transformed_img, transformed_depth, transformed_seg, cam_K, R, t, coord_K
    
def batch_data(datum, xyDim, batch_size = 5, sigma=0.2, test_mode=False):
    ld = load_data_item(datum, test_mode=test_mode)
    
    all_d = []
    for _ in range(batch_size):
        all_d.append( extract_item(ld, xyDim, sigma=sigma, test_mode=test_mode))
        
    return all_d    

def Dataset(data_, xyDim, times=1, group_size=1, random=False, sigma=0.2, test_mode=False):
    
    def gen():
        data = data_ * times
        if random:
            from random import sample
            data = sample(data, k = len(data))
        for d in data:
            all_d = batch_data(d, xyDim, batch_size=group_size, sigma=sigma, test_mode=test_mode)
            for elem in all_d:
                yield elem
#             yield extract_item(load_data_item(d))
    
    return tf.data.Dataset.from_generator(
        gen,
        tuple([_.dtype for _ in next(gen())]),
        tuple([tf.TensorShape(_.shape) for _ in next(gen())])
        )