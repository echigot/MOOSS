from pathlib import Path


# ==========================
# GENETIC ALGORITHM CONFIGURATION
# ==========================
RUN_OPTIMIZATION = True
RUN_GENERATION = True
RUN_TRAINING = True
RUN_CMMD = True
POPULATION_SIZE = 20
N_GENERATIONS = 20
N_OFFSPRINGS = POPULATION_SIZE

# ==========================
# EVALUATION CONFIGURATION
# ==========================
N_IMAGES_EVAL = 5  # Number of images to use for FID/LPIPS evaluation
N_IMAGES_TRAIN = 250  # Number of images to use for training

# ==========================
# STYLE IMAGES
# ==========================
STYLE_IMAGES = {
    "night": "./images/GP010376_frame_000129_rgb_anon.png",
    "snow": "./images/GOPR0122_frame_000234_rgb_anon.png",
    "fog": "./images/GOPR0475_frame_000049_rgb_anon.png",
    "rain": "./images/GP010400_frame_000032_rgb_anon.png",
    "day": "./images/zurich_000087_000019_leftImg8bit.png",
}

# ==========================
# AUGMENTATION CONFIGURATION
# ==========================
AUGMENTATION_LIST = [
    "stop",
    "normalize",
    "blur",
    "brighten",
    "darken",
    "contrast",
    "sharpness",
    "controlnet",
    "adain",
    "cacti"
]

AUGMENTATION_DICT = {idx: name for idx, name in enumerate(AUGMENTATION_LIST)}

AUGMENTATION_PARAMS = {
    "blur": {
        "radius": {"min": 0.5, "max": 4.0, "default": 2.0}
    },
    "brighten": {
        "factor": {"min": 1.1, "max": 2.0, "default": 1.5}
    },
    "darken": {
        "factor": {"min": 0.3, "max": 0.9, "default": 0.7}
    },
    "contrast": {
        "factor": {"min": 1.1, "max": 2.5, "default": 1.5}
    },
    "sharpness": {
        "factor": {"min": 1.0, "max": 3.0, "default": 2.0}
    },
}

# ==========================
# METRICS CONFIGURATION
# ==========================
METRICS_CONFIG = {
    "fid":
        {"enabled": False,
        "weight": 1e-2,
        "compute_fn": "compute_fid",
        "args": ["generated_dir", "real_dir"],
        "description": "Fréchet Inception Distance"
    },
    "lpips":
        {"enabled": False,
        "weight": 1.0,
        "compute_fn": "compute_lpips",
        "args": ["generated_dir", "synthetic_dir", "image_set"],
        "description": "Learned Perceptual Image Patch Similarity"
    },
    "dists":
        {"enabled": True,
        "weight": 1.0,
        "compute_fn": "compute_dists",
        "args": ["generated_dir", "image_set"],
        "description": "Deep Image Structure and Texture Similarity"
    },
    "dreamsim":
        {"enabled": True,
        "weight": 1.0,
        "compute_fn": "compute_dreamsim",
        "args": ["generated_dir", "real_dir"],
        "description": "DreamSim perceptual distance"
    },
    "dino":
        {"enabled": False,
        "weight": 1e-3,
        "compute_fn": "compute_dino_distance",
        "args": ["generated_dir", "real_dir"],
        "description": "DINO feature distance"
    },
    "cmmd":
        {"enabled": False,
        "weight": 1,
        "compute_fn": "compute_cmmd",
        "args": ["generated_dir", "real_dir"],
        "description": "CLIP Maximum Mean Discrepancy"
    }
    
}

# Get enabled metrics
ENABLED_METRICS = {k: v for k, v in METRICS_CONFIG.items() if v["enabled"]}
N_OBJECTIVES = len(ENABLED_METRICS)

# Calculer le nombre total de paramètres
N_CONTINUOUS_VARS = sum(len(params) for params in AUGMENTATION_PARAMS.values())
N_DISCRETE_VARS = len(AUGMENTATION_LIST)

# ==========================
# PATHS
# ==========================
SYNTHETIC_DIR = Path("var_home/datasets/gta/images")
REAL_DIR = Path("./images/")
string_metrics = "_".join(ENABLED_METRICS.keys())
OUTPUT_BASE_DIR = Path(f"./output/GA_{POPULATION_SIZE}pop_{N_GENERATIONS}gen_{N_IMAGES_EVAL}imgs/{string_metrics}/")