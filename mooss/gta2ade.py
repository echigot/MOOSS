import numpy as np
from PIL import Image

ade20k_palette = np.asarray([
    [0, 0, 0],
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
])

gta_classes = ['unlabeled', 'ego vehicle', 'rectification border', 'out of roi', 'static',
           'dynamic', 'ground', 'road', 'sidewalk', 'parking',
           'rail track', 'building', 'wall', 'fence', 'guard rail',
           'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light',
           'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
           'rider', 'car', 'truck', 'bus', 'caravan', 'trailer',
           'train', 'motorcycle', 'bicycle', 'license plate']
gta_class_map = {i: name for i, name in enumerate(gta_classes)}

cityscapes_classes = [
    "road",            # 0
    "sidewalk",        # 1
    "building",        # 2
    "wall",            # 3
    "fence",           # 4
    "pole",            # 5
    "traffic light",   # 6
    "traffic sign",    # 7
    "vegetation",      # 8
    "terrain",         # 9
    "sky",             # 10
    "person",          # 11
    "rider",           # 12
    "car",             # 13
    "truck",           # 14
    "bus",             # 15
    "train",           # 16
    "motorcycle",      # 17
    "bicycle",         # 18
    "unlabeled"        # 255 (usually treated as ignore index)
]

city_class_map = {i: name for i, name in enumerate(cityscapes_classes)}
city_class_map[-1] = "unlabeled"  # Add unlabeled class for Cityscapes

gta_to_ade20k_map = {
    'unlabeled': 0, 
    'ego vehicle': 0, 
    'rectification border': 0, 
    'out of roi': 0, 
    'static': 0,
    'dynamic': 0, 
    'ground': 14, 
    'road': 7, 
    'sidewalk': 12, 
    'parking': 14,
    'rail track': 39, 
    'building': 2, 
    'wall': 1, 
    'fence': 33, 
    'guard rail': 33,
    'bridge': 62, 
    'tunnel': 69, 
    'pole': 94, 
    'polegroup': 94, 
    'traffic light': 137,
    'traffic sign': 44, 
    'vegetation': 18,  # 10
    'terrain': 69,  # 14
    'sky': 3, 
    'person': 13,
    'rider': 13, 
    'car': 21, 
    'truck': 84, 
    'bus': 81, 
    'caravan': 103, 
    'trailer': 84,
    'train': 0,  # 81
    'motorcycle': 117, 
    'bicycle': 128, 
    'license plate': 0  
}

def gta_to_ade20k(gta_label):
    if not isinstance(gta_label, np.ndarray):
        gta_label = np.array(gta_label)
    if len(gta_label.shape) == 2:
        ade_label = np.zeros((*gta_label.shape, 3))
    else:
        ade_label = np.zeros(gta_label.shape)
    for id, label in enumerate(gta_classes):
        ade_label[gta_label == id] = ade20k_palette[gta_to_ade20k_map[label]]
    ade_label = Image.fromarray(ade_label.astype(np.uint8), mode='RGB')
    return ade_label


def city_to_ade20k(city_label):
    if not isinstance(city_label, np.ndarray):
        raise ValueError("Input city_label must be a NumPy array.")
    
    if len(city_label.shape) == 2:  # If the input is a single-channel label
        ade_label = np.zeros((*city_label.shape, 3), dtype=np.uint8)  # Initialize as a 3D array for RGB
    else:
        raise ValueError("Input city_label must be a single-channel label (2D array).")
    
    for id, label in enumerate(cityscapes_classes):
        if label in gta_to_ade20k_map:  # Ensure the label exists in the mapping
            ade_label[city_label == id] = ade20k_palette[gta_to_ade20k_map[label]]
    
    ade_label = Image.fromarray(ade_label, mode='RGB')
    return ade_label