"""
Simple script to apply augmentation sequence to a dataset.
"""

import argparse
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch

# Import from augmentation module
from mooss.augmentation import (
    AUGMENTATION_DICT,
    SYNTHETIC_DIR,
    STYLE_IMAGES,
    apply_augmentation,
    apply_cacti_augmentation,
    generate_augmented_dataset
)

from mooss.controlnet_utils import load_controlnet_pipeline

from CACTIF.config import RunConfig as CACTIFRunConfig
from CACTIF.cactif_model import CACTIFModel
from CACTIF.utils.latent_utils import load_or_invert_one_image
from CACTIF.utils.mask_utils import process_label
from CACTIF import palettes
from diffusers.training_utils import set_seed as cacti_set_seed


def apply_augmentation_sequence(
    aug_sequence: list,
    input_dir: Path,
    output_dir: Path,
    label_dir: Path = None,
    max_images: int = None,
    hyperparams: dict = None,
    use_sequential_styles: bool = False
):
    """
    Apply augmentation sequence to dataset.
    
    Args:
        aug_sequence: List of augmentation names (e.g., ["normalize", "blur", "controlnet"])
        input_dir: Directory with input images
        output_dir: Directory to save results
        label_dir: Directory with labels (optional)
        max_images: Max number of images to process per style
        hyperparams: Dictionary of hyperparameters for each augmentation
        use_sequential_styles: If True, apply styles in blocks (style1 for images 1-max_images, style2 for max_images+1 to 2*max_images, etc.)
    """
    # Create output directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    if label_dir:
        (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    # Get style images
    style_images_list = list(STYLE_IMAGES.values())
    
    # Get images
    all_image_paths = sorted(list(input_dir.glob("*.png")))
    
    # Determine how many images to process
    if use_sequential_styles and max_images:
        # Process max_images per style
        total_images = min(len(all_image_paths), max_images * len(style_images_list))
        image_paths = all_image_paths[:total_images]
    elif max_images:
        image_paths = all_image_paths[:max_images]
    else:
        image_paths = all_image_paths
    
    print(f"\nProcessing {len(image_paths)} images")
    print(f"Sequence: {' -> '.join(aug_sequence)}")
    if use_sequential_styles:
        images_per_style = max_images if max_images else len(image_paths) // len(style_images_list)
        print(f"Using sequential styles: {images_per_style} images per style ({len(style_images_list)} styles)")
    if hyperparams:
        print(f"Hyperparameters: {hyperparams}\n")
    
    # Load pipelines once
    pipe_control = None
    cacti_model = None
    cacti_cfg = None
    temp_dir = None
    
    if "controlnet" in aug_sequence or "adain" in aug_sequence:
        print("Loading ControlNet pipeline...")
        pipe_control = load_controlnet_pipeline()
    
    if "cacti" in aug_sequence:
        print("Loading CACTI pipeline...")
        cacti_cfg = CACTIFRunConfig()
        cacti_params = hyperparams.get("cacti", {}) if hyperparams else {}
        cacti_cfg.adain_class = cacti_params.get("adain_class", True)
        cacti_cfg.filtering = False
        
        cacti_set_seed(cacti_cfg.seed)
        cacti_model = CACTIFModel(cacti_cfg)
        cacti_model.pipe.scheduler.set_timesteps(cacti_cfg.num_timesteps)
        cacti_model.enable_edit = True
        
        # Prepare temp directory
        temp_dir = Path("/tmp/cacti_temp")
        temp_dir.mkdir(exist_ok=True)
    
    # Track current style
    current_style_idx = None
    current_reference = None
    latents_style = None
    noise_style = None
    
    # Process each image
    for img_idx, img_path in enumerate(tqdm(image_paths)):
        # Determine which style to use
        if use_sequential_styles and max_images:
            # Block-based style selection
            style_idx = img_idx // max_images
            style_idx = min(style_idx, len(style_images_list) - 1)  # Clamp to available styles
        elif use_sequential_styles:
            # If max_images not specified, divide images equally
            images_per_style = len(image_paths) // len(style_images_list)
            style_idx = min(img_idx // images_per_style, len(style_images_list) - 1)
        else:
            # Use last style by default
            style_idx = len(style_images_list) - 1
        
        # Load reference style if it changed
        if current_style_idx != style_idx:
            current_style_idx = style_idx
            reference_path = style_images_list[style_idx]
            current_reference = Image.open(reference_path).convert("RGB").resize((1024, 512))
            
            print(f"\n→ Switching to style {style_idx + 1}/{len(style_images_list)}: {Path(reference_path).stem}")
            
            # Prepare CACTI style image if needed
            if "cacti" in aug_sequence:
                # Prepare style image
                temp_style_img = temp_dir / f"style_{style_idx}.png"
                current_reference.save(temp_style_img)
                
                # Load first label as style reference
                temp_style_label = temp_dir / f"style_label_{style_idx}.png"
                first_label = sorted(list((label_dir or input_dir.parent / "labels").glob("*_labelTrainIds.png")))[0]
                if first_label.exists():
                    style_label = Image.open(first_label).convert("RGB")
                else:
                    style_label = Image.new("RGB", (1024, 512), (0, 0, 0))
                style_label.save(temp_style_label)
                
                # Process style label
                with torch.no_grad():
                    label_style, label_style_adain = process_label(temp_style_label, palettes.GTA)
                    cacti_model.label_style = [label_style]
                    cacti_model.label_style_adain = label_style_adain
                    
                    # Invert style image
                    cacti_cfg.update_latents_path(None, "style")
                    latents_style, noise_style = load_or_invert_one_image(
                        cacti_model.pipe, cacti_cfg, img_path=temp_style_img, type_img="style"
                    )
        
        # Load image
        image = Image.open(img_path).convert("RGB").resize((1024, 512))
        
        # Load label if available
        label = None
        label_path = None
        if label_dir:
            label_path = label_dir / f"{img_path.stem}_labelTrainIds.png"
            if label_path.exists():
                label = Image.open(label_path).convert("RGB")
        
        # Apply augmentations
        for aug_name in aug_sequence:
            if aug_name == "stop":
                break
            
            aug_params = hyperparams.get(aug_name, {}) if hyperparams else {}
            
            if aug_name == "cacti":
                image = apply_cacti_augmentation(
                    image, label, current_reference,
                    cacti_model, cacti_cfg, latents_style, noise_style,
                    aug_params, temp_dir
                )
            else:
                image = apply_augmentation(
                    image, label, aug_name, current_reference, aug_params, pipe_control
                )
        
        # Save
        image.save(output_dir / "images" / img_path.name)
        if label and label_path:
            label.save(output_dir / "labels" / f"{img_path.stem}_labelTrainIds.png")
    
    # Cleanup
    if cacti_model is not None and temp_dir is not None:
        for f in temp_dir.glob("*"):
            f.unlink()
    
    # Save metadata
    metadata = {
        "augmentation_sequence": aug_sequence,
        "hyperparameters": hyperparams,
        "n_images": len(image_paths),
        "input_dir": str(input_dir),
        "use_sequential_styles": use_sequential_styles
    }
    
    if use_sequential_styles:
        metadata["n_style_images"] = len(style_images_list)
        metadata["images_per_style"] = max_images if max_images else len(image_paths) // len(style_images_list)
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Done! Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Apply augmentation sequence")
    
    parser.add_argument(
        "--sequence",
        type=str,
        nargs="+",
        default=[],
        help="Augmentation sequence (e.g., normalize blur controlnet)"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(SYNTHETIC_DIR),
        help="Input images directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    
    parser.add_argument(
        "--label-dir",
        type=str,
        default=None,
        help="Labels directory (optional)"
    )
    
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max images to process (per style if --sequential-styles is used)"
    )
    
    parser.add_argument(
        "--from-json",
        type=str,
        default=None,
        help="Load sequence from eval.json or optimization_results.json"
    )
    
    parser.add_argument(
        "--solution-id",
        type=int,
        default=None,
        help="Solution ID if using optimization_results.json (None = all solutions)"
    )
    
    parser.add_argument(
        "--optimal-only",
        action="store_true",
        help="Process only optimal solution (is_optimal: true) from optimization_results.json"
    )
    
    parser.add_argument(
        "--sequential-styles",
        action="store_true",
        help="Apply styles in blocks: style1 for images 1-max_images, style2 for images max_images+1 to 2*max_images, etc."
    )
    
    args = parser.parse_args()
    
    # Load sequence(s) from JSON if specified
    sequences_to_process = []
    
    if args.from_json:
        json_path = Path(args.from_json)
        with open(json_path) as f:
            data = json.load(f)
        
        if "pareto_front" in data:
            # optimization_results.json
            if args.optimal_only:
                # Process only optimal solution
                optimal_solutions = [
                    (i, sol) for i, sol in enumerate(data["pareto_front"]["solutions"], 1)
                    if sol.get("is_optimal", False)
                ]
                if optimal_solutions:
                    solution_id, solution = optimal_solutions[0]
                    sequences_to_process.append((
                        solution_id,
                        solution["augmentation_sequence"],
                        solution.get("hyperparameters", {})
                    ))
                    print(f"Loaded optimal solution {solution_id} from {json_path}")
                else:
                    print("Error: No optimal solution found in JSON")
                    return
            elif args.solution_id is None:
                # Process all solutions
                for i, solution in enumerate(data["pareto_front"]["solutions"], 1):
                    sequences_to_process.append((
                        i,
                        solution["augmentation_sequence"],
                        solution.get("hyperparameters", {})
                    ))
                print(f"Loaded {len(sequences_to_process)} solutions from {json_path}")
            else:
                # Process specific solution
                solution = data["pareto_front"]["solutions"][args.solution_id - 1]
                sequences_to_process.append((
                    args.solution_id,
                    solution["augmentation_sequence"],
                    solution.get("hyperparameters", {})
                ))
                print(f"Loaded solution {args.solution_id} from {json_path}")
        else:
            # eval.json
            sequences_to_process.append((
                1,
                data["augmentation_sequence"],
                data.get("hyperparameters", {})
            ))
            print(f"Loaded sequence from {json_path}")
    else:
        sequences_to_process.append((1, args.sequence, {}))
    
    # Validate sequences
    valid_augs = set(AUGMENTATION_DICT.values())
    for _, sequence, _ in sequences_to_process:
        for aug in sequence:
            if aug not in valid_augs:
                print(f"Error: Invalid augmentation '{aug}'")
                print(f"Valid options: {', '.join(valid_augs)}")
                return
    
    # Process each sequence
    for solution_id, sequence, hyperparams in sequences_to_process:
        if len(sequences_to_process) > 1:
            output_dir = Path(args.output_dir) / f"solution_{solution_id}"
            print(f"\n{'='*60}")
            print(f"Processing solution {solution_id}/{len(sequences_to_process)}")
            print(f"{'='*60}")
        else:
            output_dir = Path(args.output_dir)
        
        # Apply augmentations
        # apply_augmentation_sequence(
        #     aug_sequence=sequence,
        #     input_dir=Path(args.input_dir),
        #     output_dir=output_dir,
        #     label_dir=Path(args.label_dir) if args.label_dir else Path(args.input_dir).parent / "labels",
        #     max_images=args.max_images,
        #     hyperparams=hyperparams,
        #     use_sequential_styles=args.sequential_styles
        # )
        
        generate_augmented_dataset(sequence, None, None, output_dir, args.max_images, True)


if __name__ == "__main__":
    main()