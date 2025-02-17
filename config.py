# -----------------
# DATASET ROOTS
# ----------------- 
# domainnet_dataroot = '/disk/work/hjwang/gcd/domainbed/data/domain_net/'
domainnet_dataroot = '/disk/datasets/domain_net/'

cub_root = '/disk/datasets/ood_zoo/ood_data/CUB'
cubc_root = '/disk/work/hjwang/gcd/data/cub-c'

fgvc_root = '/disk/datasets/ood_zoo/ood_data/aircraft/fgvc-aircraft-2013b'
fgvcc_root = '/disk/work/hjwang/gcd/data/fgvc-c'

scars_root = '/disk/datasets/ood_zoo/ood_data/stanford_car/cars_{}/'
scarsc_root = '/disk/work/hjwang/gcd/data/scars-c'
scars_meta_path = "/disk/datasets/ood_zoo/ood_data/stanford_car/devkit/cars_{}.mat"

# OSR Split dir
osr_split_dir = '/disk/work/hjwang/osrd/data/open_set_splits'

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path = '/disk/work/hjwang/pretrained_models/dino/dino_vitbase16_pretrain.pth' 
clip_pretrain_path = '/disk/work/hjwang/pretrained_models/clip/ViT-B-16.pt' 
feature_extract_dir = '/disk/work/hjwang/gcd/extracted_features_public_impl'     # Extract features to this directory

# -----------------
# Corruption types
# -----------------
# distortions = [
#     'gaussian_noise', 'shot_noise', 'impulse_noise',
#     'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
#     'snow', 'frost', 'fog', 'elastic_transform', 'pixelate', 'jpeg_compression',
#     'speckle_noise', 'spatter'#, 'saturate'
# ]
distortions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'zoom_blur',
    'snow', 'frost', 'fog',
    'speckle_noise', 'spatter'#, 'saturate'
]
severity = [1, 2, 3, 4, 5]

# -----------------
# domain types
# -----------------
ovr_envs = ["real", "painting", "quickdraw", "sketch", "clipart", "infograph"]