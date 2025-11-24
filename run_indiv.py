import argparse
import json
import shutil
import sys
import subprocess
import os
from pathlib import Path

# Add mooss to path
sys.path.insert(0, str(Path(__file__).parent))

from mooss.config import (
    N_IMAGES_TRAIN,
    OUTPUT_BASE_DIR,
    RUN_GENERATION,
    RUN_TRAINING,
    RUN_CMMD,
    AUGMENTATION_DICT
)

from mooss.augmentation import generate_augmented_dataset
from mooss.metrics import compute_cmmd
import tempfile


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Génération Dataset + DAFormer + CMMD')
    
    parser.add_argument(
        '--sequence',
        type=str,
        required=True,
        help='Séquence d\'augmentations séparées par virgules (ex: "darken,controlnet,adain")'
    )
    
    parser.add_argument(
        '--n-images',
        type=int,
        default=N_IMAGES_TRAIN,
        help=f'Nombre d\'images à générer (défaut: {N_IMAGES_TRAIN})'
    )
    
    return parser.parse_args()


def sequence_to_indices(aug_sequence_names: list) -> list:
    """Convert augmentation names to indices."""
    # Invert AUGMENTATION_DICT for lookup
    name_to_idx = {name: idx for idx, name in AUGMENTATION_DICT.items()}
    
    indices = []
    for name in aug_sequence_names:
        if name not in name_to_idx:
            print(f"  ❌ Augmentation inconnue: {name}")
            print(f"  → Disponibles: {list(name_to_idx.keys())}")
            sys.exit(1)
        indices.append(name_to_idx[name])
    
    return indices


def create_reference_directory(n_images_per_style: int = 100):
    """Create a temporary directory with reference images."""
    print(f"\n  → Création du répertoire de référence...")
    temp_dir = Path(tempfile.mkdtemp(prefix="cmmd_ref_"))
    
    acdc_root = Path("var_home/datasets/acdc/rgb_anon")
    styles = ['fog', 'night', 'rain', 'snow']
    
    count = 0
    for style in styles:
        style_dir = acdc_root / style / "train"
        if style_dir.exists():
            images = sorted(style_dir.rglob("*_rgb_anon.png"))[:n_images_per_style]
            for img in images:
                shutil.copy(img, temp_dir / f"{style}_{img.name}")
                count += 1
            print(f"  ✓ {style}: {len(images)} images")
    
    cityscapes_dir = Path("var_home/datasets/cityscapes/leftImg8bit/train")
    if cityscapes_dir.exists():
        images = sorted(cityscapes_dir.rglob("*_leftImg8bit.png"))[:n_images_per_style]
        for img in images:
            shutil.copy(img, temp_dir / f"day_{img.name}")
            count += 1
        print(f"  ✓ day: {len(images)} images")
    
    print(f"\n  → Total: {count} images")
    return temp_dir


def compute_and_save_cmmd(generated_dataset: Path, output_dir: Path, n_ref: int = 100):
    """Compute CMMD and save results."""
    print("\n  → Calcul CMMD...")
    ref_dir = create_reference_directory(n_ref)
    
    try:
        gen_dir = generated_dataset / "images"
        cmmd_score = compute_cmmd(gen_dir, ref_dir)
        
        results = {"cmmd": cmmd_score}
        with open(output_dir / "cmmd_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ CMMD: {cmmd_score:.4f}")
        return cmmd_score
        
    finally:
        shutil.rmtree(ref_dir)


def update_daformer_config(dataset_path: Path):
    """Update DAFormer config with new dataset path."""
    config_file = Path("var_home/DAFormer/configs/_base_/datasets/uda_gta_to_cityscapes_512x512.py")
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")
    
    backup_file = config_file.with_suffix('.py.backup')
    if not backup_file.exists():
        shutil.copy(config_file, backup_file)
    
    with open(config_file, 'r') as f:
        lines = f.readlines()
    
    in_target_section = False
    indent_level = 0
    updated_lines = []
    
    for line in lines:
        if 'target=dict(' in line:
            in_target_section = True
            indent_level = 1
            updated_lines.append(line)
            continue
        
        if in_target_section:
            indent_level += line.count('(') - line.count(')')
            
            if 'data_root' in line and '=' in line:
                indent = len(line) - len(line.lstrip())
                line = ' ' * indent + f"data_root='{dataset_path.absolute()}/',\n"
            elif 'img_dir' in line and '=' in line:
                indent = len(line) - len(line.lstrip())
                line = ' ' * indent + "img_dir='images',\n"
            elif 'ann_dir' in line and '=' in line:
                indent = len(line) - len(line.lstrip())
                line = ' ' * indent + "ann_dir='labels',\n"
            
            updated_lines.append(line)
            
            if indent_level == 0:
                in_target_section = False
        else:
            updated_lines.append(line)
    
    with open(config_file, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"  ✓ Config updated: {dataset_path.absolute()}")
    return config_file


def run_daformer_training():
    """Launch DAFormer training."""
    daformer_dir = Path("var_home/DAFormer")
    print("\n  → Lancement entraînement DAFormer...")
    
    original_dir = Path.cwd()
    os.chdir(daformer_dir)
    
    try:
        cmd = ["conda", "run", "-n", "daformer", "python", "run_experiments.py", "--exp", "11"]
        subprocess.run(cmd, check=True)
        print("  ✓ Entraînement terminé")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Échec: {e}")
        return False
    finally:
        os.chdir(original_dir)


def main():
    args = parse_arguments()
    
    # Parse sequence
    aug_names = [name.strip() for name in args.sequence.split(',')]
    print(f"\n{'='*60}")
    print("PIPELINE: Génération + DAFormer + CMMD")
    print(f"{'='*60}")
    print(f"Séquence: {' -> '.join(aug_names)}")
    print(f"Nombre d'images: {args.n_images}")
    
    # Convert to indices
    aug_indices = sequence_to_indices(aug_names)
    print(f"Indices: {aug_indices}")
    
    # Output directory
    dataset_output = OUTPUT_BASE_DIR / f"dataset_{'_'.join(aug_names[:3])}"
    dataset_output.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # STEP 1: GENERATE DATASET
    # ========================================
    print(f"\n[1/3] Génération du dataset...")
    
    try:
        if RUN_GENERATION:
            generate_augmented_dataset(
                aug_indices, 
                None, 
                None, 
                dataset_output, 
                args.n_images, 
                use_sequential_styles=True
            )
        
        # Create labels symlink
        labels_symlink = dataset_output / "labels"
        label_dir = Path("var_home/datasets/gta_crop/labels")
        
        if not labels_symlink.exists():
            labels_symlink.symlink_to(label_dir)
            print(f"  ✓ Labels liés: {label_dir}")
        
        n_generated = len(list((dataset_output / "images").glob("*.png")))
        print(f"  ✓ {n_generated} images générées dans {dataset_output}")
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================
    # STEP 2: DAFORMER TRAINING
    # ========================================
    print(f"\n[2/3] Entraînement DAFormer...")
    
    try:
        update_daformer_config(dataset_output)
        
        if RUN_TRAINING:
            training_success = run_daformer_training()
            if not training_success:
                print("  ⚠️  Entraînement échoué, continuation...")
        else:
            print("  ⏭️  Entraînement ignoré (RUN_TRAINING=False)")
            
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
    
    # ========================================
    # STEP 3: CMMD EVALUATION
    # ========================================
    print(f"\n[3/3] Évaluation CMMD...")
    
    cmmd_score = None
    if RUN_CMMD:
        try:
            cmmd_score = compute_and_save_cmmd(dataset_output, dataset_output, args.n_images)
        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ⏭️  CMMD ignoré (RUN_CMMD=False)")
    
    # ========================================
    # SUMMARY
    # ========================================
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_output}")
    print(f"Images: {n_generated}")
    if cmmd_score:
        print(f"CMMD: {cmmd_score:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrompu")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)