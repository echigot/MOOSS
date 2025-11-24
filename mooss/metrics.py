import sys
sys.path.insert(0, 'var_home/cmmd-pytorch')

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.dists import DeepImageStructureAndTextureSimilarity
from dreamsim import dreamsim
from tqdm import tqdm

from .config import STYLE_IMAGES, N_IMAGES_EVAL


def load_images_as_tensors(image_dir: Path, max_images: int = None) -> torch.Tensor:
    """Load images from directory as tensor [N, 3, H, W]."""
    image_paths = sorted(list(image_dir.rglob("*.png")))
    if max_images:
        image_paths = image_paths[:max_images]
    
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((1024, 512))
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        images.append(img_tensor)
    return torch.stack(images)


def load_specific_images_as_tensors(image_paths: list) -> torch.Tensor:
    """Load specific images as tensor [N, 3, H, W]."""
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((1024, 512))
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        images.append(img_tensor)
    return torch.stack(images)


def compute_fid(generated_dir: Path, real_dir: Path) -> float:
    """Compute FID between generated and real images."""
    fid = FrechetInceptionDistance(normalize=True)
    
    generated_images = load_images_as_tensors(generated_dir, N_IMAGES_EVAL)
    real_images = load_images_as_tensors(real_dir, 200)
    
    generated_images = (generated_images * 255).to(torch.uint8)
    real_images = (real_images * 255).to(torch.uint8)
    
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    
    return fid.compute().item()


def compute_lpips(generated_dir: Path, synthetic_dir: Path, image_set: list) -> float:
    """Compute LPIPS between generated and original synthetic images."""
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True)
    
    generated_paths = [generated_dir / img_path.name for img_path in image_set]
    generated_images = load_specific_images_as_tensors(generated_paths)
    synthetic_images = load_specific_images_as_tensors(image_set)
    
    lpips_scores = []
    for gen_img, syn_img in zip(generated_images, synthetic_images):
        score = lpips(gen_img.unsqueeze(0), syn_img.unsqueeze(0))
        lpips_scores.append(score.item())
    
    if np.mean(lpips_scores) < 1e-3:
        return 1.0
    return np.mean(lpips_scores)

def compute_dists(generated_dir: Path, image_set:list) -> float:
    """Compute DISTS between generated and original synthetic images."""
    dists = DeepImageStructureAndTextureSimilarity()
    
    generated_paths = [generated_dir / img_path.name for img_path in image_set]
    generated_images = load_specific_images_as_tensors(generated_paths)
    synthetic_images = load_specific_images_as_tensors(image_set)
    
    dists_scores = []
    for gen_img, syn_img in tqdm(zip(generated_images, synthetic_images)):
        score = dists(gen_img.unsqueeze(0), syn_img.unsqueeze(0))
        dists_scores.append(score.item())
    
    if np.mean(dists_scores) < 1e-3:
        return 1.0
    return np.mean(dists_scores)



def compute_dreamsim(generated_dir: Path, real_image: Path) -> float:
    """Compute DreamSim distance between generated images and reference real images."""
    
    model, preprocess = dreamsim(pretrained=True, cache_dir="var_home/.cache")
    model = model.to("cuda")
    
    distances = []
    generated_paths = sorted(list(generated_dir.rglob("*.png")))
    
    for gen_path, reference in zip(generated_paths, STYLE_IMAGES.values()):
        gen_img = preprocess(Image.open(gen_path)).to("cuda")
        ref_img = preprocess(Image.open(reference)).to("cuda")
        with torch.no_grad():
            distance = model(ref_img, gen_img)
            distances.append(distance.item())
    
    return np.mean(distances)


def compute_dreamsim_full_dataset(generated_dir: Path, references_path: Path) -> float:
    """Compute DreamSim distance between generated images and reference real images."""
    
    model, preprocess = dreamsim(pretrained=True, cache_dir="var_home/.cache")
    model = model.to("cuda")
    
    distances = []
    generated_paths = sorted(list(generated_dir.rglob("*.png")))
    
    for gen_path, reference in zip(generated_paths, references_path):
        gen_img = preprocess(Image.open(gen_path)).to("cuda")
        ref_img = preprocess(Image.open(reference)).to("cuda")
        with torch.no_grad():
            distance = model(ref_img, gen_img)
            distances.append(distance.item())
    
    return np.mean(distances)


def compute_dino_distance(generated_dir: Path, real_dir: Path) -> float:
    """Compute DINO feature distance between generated and real images."""
    import sys
    sys.path.insert(0, 'var_home/dgm-eval')
    from dgm_eval.models import load_encoder
    from dgm_eval.dataloaders import get_dataloader
    from dgm_eval.representations import get_representations
    from dgm_eval.metrics import compute_FD_with_reps
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_encoder(
        'dinov2', 
        device, 
        ckpt=None, 
        arch=None,
        clean_resize=True,
        sinception=False,
        depth=0
    )
    
    dataloader_real = get_dataloader(
        str(real_dir), 
        nsample=200,
        batch_size=50, 
        num_workers=4,
        seed=42,
        sample_w_replacement=False,
        transform=lambda x: model.transform(x)
    )
    
    dataloader_gen = get_dataloader(
        str(generated_dir),
        nsample=N_IMAGES_EVAL,
        batch_size=50,
        num_workers=4,
        seed=42,
        sample_w_replacement=False,
        transform=lambda x: model.transform(x)
    )
    
    print("  Computing DINO representations for real images...")
    reps_real = get_representations(model, dataloader_real, device, normalized=False)
    
    print("  Computing DINO representations for generated images...")
    reps_gen = get_representations(model, dataloader_gen, device, normalized=False)
    
    fd_dino = compute_FD_with_reps(reps_real, reps_gen)
    
    return fd_dino


def compute_cmmd(generated_dir: Path, real_dir: Path) -> float:
    """Compute CMMD between generated and real images."""
    from .cmmd import (
        ClipEmbeddingModel, 
        compute_embeddings_for_dir, 
        mmd
    )
    
    embedding_model = ClipEmbeddingModel()
    batch_size = 32
    max_count = -1
    
    real_embs = compute_embeddings_for_dir(
        real_dir, embedding_model, batch_size, max_count
    )
    generated_embs = compute_embeddings_for_dir(
        generated_dir, embedding_model, batch_size, max_count
    )
    
    cmmd_value = mmd(real_embs, generated_embs)
    return cmmd_value.item() if torch.is_tensor(cmmd_value) else float(cmmd_value)


class MetricsEvaluator:
    """Class to handle metric computation and management."""
    
    def __init__(self, metrics_config: dict):
        self.metrics_config = metrics_config
        self.metrics_functions = {
            "compute_fid": compute_fid,
            "compute_lpips": compute_lpips,
            "compute_dists": compute_dists,
            "compute_dreamsim": compute_dreamsim,
            "compute_dino_distance": compute_dino_distance,
            "compute_cmmd": compute_cmmd
        }
    
    def evaluate_all(self, generated_dir: Path, real_dir: Path, 
                     synthetic_dir: Path = None, image_set: list = None) -> dict:
        """Evaluate all enabled metrics."""
        results = {}
        
        print("\nComputing metrics...")
        for metric_name, metric_config in self.metrics_config.items():
            if not metric_config["enabled"]:
                continue
            
            print(f"  Computing {metric_config['description']}...")
            
            compute_fn = self.metrics_functions[metric_config["compute_fn"]]
            
            args = []
            for arg_name in metric_config["args"]:
                if arg_name == "generated_dir":
                    args.append(generated_dir)
                elif arg_name == "real_dir":
                    args.append(real_dir)
                elif arg_name == "synthetic_dir":
                    args.append(synthetic_dir)
                elif arg_name == "image_set":
                    args.append(image_set)
            
            try:
                value = compute_fn(*args)
                results[metric_name] = value * metric_config["weight"]
                print(f"    {metric_name.upper()}: {value:.4f}")
            except Exception as e:
                print(f"    Error computing {metric_name}: {e}")
                results[metric_name] = float('inf')
                exit(0)
        
        return results
    
    def get_objective_array(self, results: dict) -> list:
        """Convert results dictionary to array for optimization."""
        return [results[metric_name] for metric_name in self.metrics_config.keys() 
                if self.metrics_config[metric_name]["enabled"]]
    
    def get_metric_names(self) -> list:
        """Get list of enabled metric names."""
        return [name for name, config in self.metrics_config.items() if config["enabled"]]