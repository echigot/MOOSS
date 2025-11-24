import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

from .adain import adain
from .config import (
    AUGMENTATION_DICT, 
    AUGMENTATION_PARAMS,
    STYLE_IMAGES,
    SYNTHETIC_DIR,
    N_DISCRETE_VARS,
    N_IMAGES_EVAL
)
from .controlnet_utils import (
    normalize_with_reference,
    convert_label_to_ade,
    generate_canny_image,
    run_inference,
    load_controlnet_pipeline,
)

from CACTIF import palettes
from CACTIF.config import RunConfig as CACTIFRunConfig, Range
from CACTIF.cactif_model import CACTIFModel
from CACTIF.utils.latent_utils import load_or_invert_one_image, get_init_latents_and_noises
from CACTIF.utils.mask_utils import process_label

from diffusers.training_utils import set_seed as cacti_set_seed


def extract_hyperparameters(hyperparams_flat: np.ndarray) -> dict:
    """
    Convert flat array of hyperparameters to structured dictionary.
    
    Args:
        hyperparams_flat: Flat array of all hyperparameters
        
    Returns:
        Dictionary mapping augmentation names to their parameters
    """
    result = {}
    
    if len(hyperparams_flat) == 0:
        # Use default values from AUGMENTATION_PARAMS
        for aug_name, params_config in AUGMENTATION_PARAMS.items():
            result[aug_name] = {}
            for param_name, bounds in params_config.items():
                result[aug_name][param_name] = bounds["default"]
        return result
    
    idx = 0
    for aug_name, params_config in AUGMENTATION_PARAMS.items():
        result[aug_name] = {}
        for param_name, bounds in params_config.items():
            result[aug_name][param_name] = hyperparams_flat[idx]
            idx += 1
    
    return result


def apply_augmentation(image: Image, label: Image, augmentation_name: str, 
                      reference: Image, params: dict = None, pipe=None):
    """
    Apply augmentation pipeline to a single image.
    
    Args:
        image: Input image (PIL Image)
        label: Segmentation label image (PIL Image)
        augmentation_name: Name of augmentation to apply
        reference: Reference image for style transfer
        params: Dictionary of hyperparameters for the augmentation
        pipe: ControlNet pipeline (optional, passed to avoid reloading)
    
    Returns:
        Augmented image (PIL Image)
    """
    if params is None:
        params = {}
    
    if augmentation_name == "controlnet":
        canny_image = generate_canny_image(image)
        ade_label = convert_label_to_ade(label)
        
        if pipe is None:
            pipe = load_controlnet_pipeline()
        
        output_images = run_inference(
            pipe,
            reference,
            ade_label,
            canny_image,
            style_name=""
        )
        image = output_images[0] if isinstance(output_images, list) else output_images
    
    elif augmentation_name == "normalize":
        image = normalize_with_reference(image, reference)
        
    elif augmentation_name == "blend":
        alpha = params.get("alpha", 0.5)
        image = Image.blend(image, reference.resize(image.size), alpha=alpha)
        
    elif augmentation_name == "blur":
        radius = params.get("radius", 2.0)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    elif augmentation_name == "brighten":
        factor = params.get("factor", 1.5)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
        
    elif augmentation_name == "darken":
        factor = params.get("factor", 0.7)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factor)
        
    elif augmentation_name == "contrast":
        factor = params.get("factor", 1.5)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factor)
    
    elif augmentation_name == "sharpness":
        factor = params.get("factor", 2.0)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(factor)
    
    elif augmentation_name == "adain":
        if pipe is None:
            pipe = load_controlnet_pipeline()
        
        with torch.no_grad():
            style_image_latents = pipe.image_processor.preprocess(
                reference, width=1024, height=512
            ).to(dtype=torch.float16, device="cuda:0")
            style_latents = pipe.vae.encode(style_image_latents).latent_dist.mode()
            style_latents = pipe.vae.config.scaling_factor * style_latents
            
            content_image_latents = pipe.image_processor.preprocess(
                image, width=1024, height=512
            ).to(dtype=torch.float16, device="cuda:0")
            content_latents = pipe.vae.encode(content_image_latents).latent_dist.mode()
            content_latents = pipe.vae.config.scaling_factor * content_latents
            
            mixed_latents = adain(content_latents[0], style_latents[0])
            
            decoded = pipe.vae.decode(
                mixed_latents.unsqueeze(0) / pipe.vae.config.scaling_factor, 
                return_dict=False
            )[0]
            
            image = pipe.image_processor.postprocess(
                decoded, 
                output_type="pil", 
                do_denormalize=[True]
            )[0]
        
    return image


def generate_augmented_dataset(solution: list, generation_id: int, individual_id: int, 
                               output_base_dir: Path, n_images: int = N_IMAGES_EVAL,
                               use_sequential_styles: bool = False) -> tuple:
    """
    Generate a full augmented dataset with given parameters.
    
    Args:
        X: Parameter vector (augmentation sequence + hyperparameters)
        generation_id: Generation ID for output organization
        individual_id: Individual ID for output organization
        output_base_dir: Base directory for outputs
        n_images: Number of images to process (per style if use_sequential_styles=True)
        use_sequential_styles: If True, generates n_images per style (for final generation).
                               If False, distributes styles across n_images (for optimization).
    """
    solution = np.array(solution)
    sequence = solution[:N_DISCRETE_VARS].astype(int)
    hyperparams_flat = solution[N_DISCRETE_VARS:]
    hyperparams = extract_hyperparameters(hyperparams_flat)
    
    if generation_id is not None and individual_id is not None:
        output_dir = output_base_dir / f"gen_{generation_id}" / f"ind_{individual_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = output_base_dir
    
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    label_base_dir = SYNTHETIC_DIR.parent / "labels"
    
    image_paths = sorted(list(SYNTHETIC_DIR.glob("*.png")))
    
    # Get style images list
    style_images_list = list(STYLE_IMAGES.values())
    n_styles = len(style_images_list)
    
    # Determine which images to process based on mode
    if use_sequential_styles:
        # Final generation: n_images per style
        total_images = n_images * n_styles
        image_set = image_paths[:total_images]
        print(f"Sequential styles mode: {n_images} images × {n_styles} styles = {total_images} total images")
    else:
        # Optimization: distribute styles across n_images
        image_set = image_paths[3:3+n_images]
        print(f"Optimization mode: {n_images} images with styles distributed")
    
    # Load pipelines once at the beginning
    pipe_control = None
    cacti_model = None
    cacti_cfg = None
    
    if any(AUGMENTATION_DICT[aug_idx] in ["controlnet", "adain"] for aug_idx in sequence):
        print("Loading ControlNet pipeline...")
        pipe_control = load_controlnet_pipeline()
    
    if any(AUGMENTATION_DICT[aug_idx] == "cacti" for aug_idx in sequence):
        print("Loading CACTI pipeline...")
        cacti_cfg = CACTIFRunConfig()
        cacti_cfg.adain_class = hyperparams.get("cacti", {}).get("adain_class", True)
        cacti_cfg.filtering = False
        
        cacti_set_seed(cacti_cfg.seed)
        cacti_model = CACTIFModel(cacti_cfg)
        cacti_model.pipe.scheduler.set_timesteps(cacti_cfg.num_timesteps)
        cacti_model.enable_edit = True
        
    
    # Track current style
    current_style_idx = None
    style_img_path = Path("./latents")
    
    # Process each image
    for img_idx, content_img_path in enumerate(tqdm(image_set, desc=f"Gen{generation_id}_Ind{individual_id}")):
        # Determine which style to use
        if use_sequential_styles:
            # Sequential mode: style1 for images 0-n_images-1, style2 for n_images-2*n_images-1, etc.
            style_idx = img_idx // n_images
        else:
            # Optimization mode: distribute styles evenly
            # If n_styles=2 and n_images=3: style 0, 0, 1 (indices 0, 1, 2)
            images_per_style = max(1, n_images // n_styles)
            style_idx = min(img_idx // images_per_style, n_styles - 1)
        
        # Load reference style if it changed
        if current_style_idx != style_idx:
            current_style_idx = style_idx
            style_img_path = Path(style_images_list[style_idx])

            # print(f"\n→ Switching to style {style_idx + 1}/{n_styles}: {reference_path.stem}")
            
            # Prepare CACTI style resources if needed
            if "cacti" in [AUGMENTATION_DICT[aug_idx] for aug_idx in sequence]:
                # Load first label as style reference
                style_label_name = style_img_path.stem.replace("_rgb_anon", "_gt_labelColor")
                style_label_name = style_label_name.replace("_leftImg8bit", "_gtFine_color") + ".png"
                style_label_path = style_img_path.parent.parent / "labels" / style_label_name 
        

    
        output_path = images_dir / content_img_path.name
        
        content_label_name = content_img_path.stem + ".png"
        content_label_path = label_base_dir / content_label_name
        if not content_label_path.exists():
            print(f"Warning: Label not found for {content_img_path.name}, skipping...")
        
        style_image = Image.open(style_img_path).convert("RGB").resize((1024, 512), Image.NEAREST)
        image = Image.open(content_img_path).convert("RGB").resize((1024, 512), Image.NEAREST)
        label = Image.open(content_label_path).convert("RGB")
        
        for aug_idx in sequence:
            augmentation_name = AUGMENTATION_DICT[aug_idx]
            if augmentation_name == "stop":
                break
            
            aug_params = hyperparams.get(augmentation_name, {})
            
            # Passer les modèles chargés à apply_augmentation
            if augmentation_name == "cacti":
                image.save("temp_img.png")
                image = apply_cacti_augmentation(
                    cacti_model, cacti_cfg, 
                    style_img_path, style_label_path,
                    Path("temp_img.png"), content_label_path
                )
            else:
                image = apply_augmentation(
                    image, label, augmentation_name, 
                    style_image, aug_params, pipe_control
                )
        
        image.save(output_path)

    
    return output_dir, image_set


def apply_cacti_augmentation(model, cfg, 
                             style_img_path, style_label_path,
                             content_img_path, content_label_path):
    """
    Apply CACTI augmentation to a single image following the pattern from run.py.
    """
        
    with torch.no_grad():
        # Traiter les labels de style et de contenu
        label_style, label_style_adain = process_label(style_label_path, palettes.CITYSCAPES)
        model.label_style = [label_style]  # Réinitialiser pour chaque image
        model.label_style_adain = label_style_adain
        
        label_content, label_content_adain = process_label(content_label_path, palettes.GTA)
        model.label_content = label_content
        model.label_content_adain = label_content_adain
        

        # Mettre à jour les chemins de latents pour cette paire (content, style)
        cfg.update_latents_path(None, style_img_path.stem)

        # Inverser les images (ou charger les latents si déjà calculés)
        latents_style, noise_style = load_or_invert_one_image(
            model.pipe, cfg, img_path=style_img_path, type_img="style"
        )
        latents_content, noise_content = load_or_invert_one_image(
            model.pipe, cfg, img_path=content_img_path, type_img="content"
        )
        
        # Mettre à jour les latents et le bruit
        model.set_latents(latents_style, latents_content)
        model.set_noise(noise_style, noise_content)
        model.set_onehot_masks()
        
        # Obtenir les latents initiaux et les bruits
        init_latents, init_zs = get_init_latents_and_noises(model=model, cfg=cfg)
        model.pipe.scheduler.set_timesteps(cfg.num_timesteps)
        
        start_step = min(cfg.cross_attn_32_range.start, cfg.cross_attn_64_range.start)
        end_step = max(cfg.cross_attn_32_range.end, cfg.cross_attn_64_range.end)
        
        # Exécuter le transfert de style
        images = model.pipe(
            prompt=[cfg.prompt] * 3,
            latents=init_latents,
            guidance_scale=1.0,
            num_inference_steps=cfg.num_timesteps,
            swap_guidance_scale=cfg.swap_guidance_scale,
            callback=model.get_adain_callback() if cfg.adain_class else None,
            eta=1,
            zs=init_zs,
            generator=torch.Generator('cuda').manual_seed(cfg.seed),
            cross_image_attention_range=Range(start=start_step, end=end_step)
        ).images
        
        # Obtenir l'image transférée (premier élément)
        return images[0]