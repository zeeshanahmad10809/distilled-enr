# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#############################################################

# Usage: python dataset_preprocessing/shapenet/preprocess_cars_cameras.py --source ~/downloads/cars_train --dest /data/cars_preprocessed

#############################################################

"""
Preprcessing ShapeNet Cars:
File modified from the source: https://github.com/NVlabs/eg3d/blob/main/dataset_preprocessing/shapenet_cars/preprocess_shapenet_cameras.py
"""


import json
import numpy as np
import os
import shutil
from tqdm import tqdm
import argparse

def list_recursive(folderpath):
    return [os.path.join(folderpath, filename) for filename in os.listdir(folderpath)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    # parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()
    # args = {"source": "cars_train1", "max_images": None}

    # Parse cameras
    # dataset_path = args['source']
    dataset_path = args.source
    for scene_folder_path in tqdm(list_recursive(dataset_path), desc="Preprocessing scenes", total=len(list_recursive(dataset_path))):
        if not os.path.isdir(scene_folder_path): continue
        
        cameras = dict()
        for rgb_path in list_recursive(os.path.join(scene_folder_path, 'rgb')):
            relative_path = os.path.relpath(rgb_path, dataset_path)
            intrinsics_path = os.path.join(scene_folder_path, 'intrinsics.txt')
            pose_path = rgb_path.replace('rgb', 'pose').replace('png', 'txt')
            assert os.path.isfile(rgb_path)
            assert os.path.isfile(intrinsics_path)
            assert os.path.isfile(pose_path)
            
            with open(pose_path, 'r') as f:
                pose = np.array([float(n) for n in f.read().split(' ')]).reshape(4, 4).tolist()
                
            with open(intrinsics_path, 'r') as f:
                first_line = f.read().split('\n')[0].split(' ')
                focal = float(first_line[0]) 
                cx = float(first_line[1])
                cy = float(first_line[2])
                            
                orig_img_size = 512  # cars_train has intrinsics corresponding to image size of 512 * 512
                intrinsics = np.array(
                    [[focal / orig_img_size, 0.00000000e+00, cx / orig_img_size],
                    [0.00000000e+00, focal / orig_img_size, cy / orig_img_size],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
                ).tolist()
            
            cameras[rgb_path.split('/')[-1]] = np.concatenate([np.asarray(pose).reshape(-1), np.asarray(intrinsics).reshape(-1)]).tolist()
            try:
                shutil.move(rgb_path, os.path.join(scene_folder_path, rgb_path.split("/")[-1]))
            except FileNotFoundError:
                pass
        
        shutil.rmtree(os.path.join(scene_folder_path, 'rgb'))
        shutil.rmtree(os.path.join(scene_folder_path, 'pose'))
        shutil.rmtree(os.path.join(scene_folder_path, 'intrinsics'))
        os.remove(os.path.join(scene_folder_path, 'intrinsics.txt'))

        # let's sort the dictionary before the json dump
        cameras = dict(sorted(cameras.items(), key=lambda item: int(item[0].split('.')[0])))
        with open(os.path.join(scene_folder_path, 'render_params.json'), 'w') as outfile:
            json.dump(cameras, outfile, indent=4)

