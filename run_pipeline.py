import json
import shutil
import sys
import subprocess
import os
from pathlib import Path

# Add mooss to path
sys.path.insert(0, str(Path(__file__).parent))

from mooss.config import (
    POPULATION_SIZE,
    N_GENERATIONS,
    N_IMAGES_TRAIN,
    N_IMAGES_EVAL,
    OUTPUT_BASE_DIR,
    ENABLED_METRICS,
    SYNTHETIC_DIR,
    RUN_TRAINING,
    RUN_OPTIMIZATION,
    RUN_GENERATION,
    RUN_CMMD,
    STYLE_IMAGES
)

from mooss.optimization import run_optimization
from mooss.augmentation import generate_augmented_dataset

from mooss.metrics import compute_cmmd
import tempfile
import shutil

def create_cmmd_directory(n_images_per_style):
    """Create a temporary directory with reference images."""
    
    print(f"\n  → Création du répertoire de référence...")
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="cmmd_ref_"))
    
    # ACDC styles
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
    
    # Cityscapes (day)
    cityscapes_dir = Path("var_home/datasets/cityscapes/leftImg8bit/train")
    if cityscapes_dir.exists():
        images = sorted(cityscapes_dir.rglob("*_leftImg8bit.png"))[:n_images_per_style]
        for img in images:
            shutil.copy(img, temp_dir / f"day_{img.name}")
            count += 1
        print(f"  ✓ day: {len(images)} images")
    
    print(f"\n  → Total: {count} images dans {temp_dir}")
    return temp_dir


def compute_and_save_cmmd(generated_dataset: Path, output_dir: Path, n_ref: int = 100):
    """Compute CMMD and save results."""
    print("\n  → Calcul CMMD...")
    
    # Create reference directory
    ref_dir = create_cmmd_directory(n_ref)
    
    try:
        # Get generated images directory
        gen_dir = generated_dataset / "images"
        
        # Compute CMMD (using existing function without modification)
        cmmd_score = compute_cmmd(gen_dir, ref_dir)
        
        # Save results
        results = {"cmmd": cmmd_score}
        with open(output_dir / "cmmd_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ CMMD: {cmmd_score:.4f}")
        return cmmd_score
        
    finally:
        # Clean up temporary directory
        shutil.rmtree(ref_dir)
        print(f"  ✓ Répertoire temporaire supprimé")
        
        
def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_step(step_num, total_steps, description):
    """Print a step header."""
    print(f"\n[{step_num}/{total_steps}] {description}...")


def load_optimization_results(results_file: Path):
    """Load and parse optimization results."""
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    with open(results_file) as f:
        data = json.load(f)
    
    return data


def get_optimal_solution(data: dict):
    """
    Get optimal solution from results.
    
    Returns:
        tuple: (solution_id, solution_dict, hyperparameters) or (None, None, None)
    """
    solutions = data.get('pareto_front', {}).get('solutions', [])
    
    # Find optimal solution
    for i, sol in enumerate(solutions, 1):
        if sol.get('is_optimal', False):
            return i, sol['augmentation_sequence'], sol.get('hyperparameters', {}), sol['augmentation_indices']
    
    # No optimal, return best solution (first one)
    if solutions:
        return 1, solutions[0]['augmentation_sequence'], solutions[0].get('hyperparameters', {}), solutions[0]['augmentation_indices']
    
    return None, None, None, None


def update_daformer_config(dataset_path: Path):
    """
    Update DAFormer configuration with new dataset path.
    Keeps GTA as source, uses generated dataset as target.
    """
    config_file = Path("var_home/DAFormer/configs/_base_/datasets/uda_gta_to_cityscapes_512x512.py")
    
    if not config_file.exists():
        raise FileNotFoundError(f"DAFormer config not found: {config_file}")
    
    # Create backup
    backup_file = config_file.with_suffix('.py.backup')
    if not backup_file.exists():
        shutil.copy(config_file, backup_file)
        print(f"  ✓ Backup created: {backup_file}")
    
    # Read config
    with open(config_file, 'r') as f:
        lines = f.readlines()
    
    # Find and update target section
    in_target_section = False
    indent_level = 0
    updated_lines = []
    
    for line in lines:
        # Detect target=dict( line
        if 'target=dict(' in line:
            in_target_section = True
            indent_level = 1
            updated_lines.append(line)
            continue
        
        if in_target_section:
            # Count parentheses to track nesting
            indent_level += line.count('(') - line.count(')')
            
            # Update data_root
            if 'data_root' in line and '=' in line:
                indent = len(line) - len(line.lstrip())
                line = ' ' * indent + f"data_root='{dataset_path.absolute()}/',\n"
            
            # Update img_dir
            elif 'img_dir' in line and '=' in line:
                indent = len(line) - len(line.lstrip())
                line = ' ' * indent + "img_dir='images',\n"
            
            # Update ann_dir
            elif 'ann_dir' in line and '=' in line:
                indent = len(line) - len(line.lstrip())
                line = ' ' * indent + "ann_dir='labels',\n"
            
            updated_lines.append(line)
            
            # Exit target section when parentheses are balanced
            if indent_level == 0:
                in_target_section = False
        else:
            updated_lines.append(line)
    
    # Write updated config
    with open(config_file, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"  ✓ Config updated: {config_file}")
    print(f"  ✓ Source (GTA original): data/gta/")
    print(f"  ✓ Target (Generated): {dataset_path.absolute()}/")
    
    return config_file


def run_daformer_training(work_dir: Path):
    """Launch DAFormer training."""
    daformer_dir = Path("var_home/DAFormer")
    
    print("\n  → Launching DAFormer training...")
    print(f"  → Working directory: {work_dir}")
    
    # Change to DAFormer directory
    original_dir = Path.cwd()
    os.chdir(daformer_dir)
    
    try:
        # Run training with conda
        cmd = ["conda", "run", "-n", "daformer", "python", "run_experiments.py", "--exp", "11"]
        result = subprocess.run(cmd, check=True)
        
        print("  ✓ Training completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  Training failed: {e}")
        return False
        
    finally:
        # Restore directory
        os.chdir(original_dir)


def main():
    """Run the complete pipeline."""
    
    # Print configuration
    print_header("PIPELINE COMPLET: GA + GÉNÉRATION + DAFORMER + ÉVALUATION ACDC")
    print(f"Population: {POPULATION_SIZE}")
    print(f"Générations: {N_GENERATIONS}")
    print(f"Images évaluation: {N_IMAGES_EVAL}")
    print(f"Images entraînement: {N_IMAGES_TRAIN}")
    print(f"Répertoire de sortie: {OUTPUT_BASE_DIR}")
    print("\nMétriques actives:")
    for metric, config in ENABLED_METRICS.items():
        print(f"  - {metric}: {config['description']}")
    
    # ========================================
    # STEP 1: GENETIC ALGORITHM OPTIMIZATION
    # ========================================
    if RUN_OPTIMIZATION:
        print_step(1, 5, "Optimisation par algorithme génétique")
        results_dir = run_optimization()
        print("  ✓ Optimisation terminée")
        print(f"  ✓ Résultats sauvegardés: {results_dir}")
    else:
        print("\n[1/5] Optimisation GA - IGNORÉE")

    
    # ========================================
    # STEP 2: LOAD RESULTS
    # ========================================
    print_step(2, 4, "Chargement des résultats")
    
    results_file = OUTPUT_BASE_DIR / "optimization_results.json"
    
    try:
        data = load_optimization_results(results_file)
        solution_id, aug_sequence, hyperparams, aug_indices = get_optimal_solution(data)
        
        if solution_id is None:
            print("  Aucune solution trouvée")
            sys.exit(1)
        
        if data.get('pareto_front', {}).get('solutions', [])[solution_id - 1].get('is_optimal', False):
            print(f"  ✓ Solution optimale trouvée (ID: {solution_id})")
        else:
            print(f"  Utilisation de la meilleure solution (ID: {solution_id})")
        
        print(f"  → Séquence: {' -> '.join(aug_sequence)}")
        if hyperparams:
            print(f"  → Hyperparamètres: {hyperparams}")
        
    except Exception as e:
        print(f"  Erreur lors du chargement: {e}")
        sys.exit(1)
    
    # ========================================
    # STEP 3: GENERATE DATASET
    # ========================================
    print_step(3, 4, "Génération du dataset")
    
    dataset_output = OUTPUT_BASE_DIR / "dataset_mooss"
    label_dir = SYNTHETIC_DIR.parent / "labels"
    
    try:
        if RUN_GENERATION:
            generate_augmented_dataset(aug_indices, None, None, dataset_output, N_IMAGES_TRAIN, True)
        else:
            print("\n[3/5] Génération dataset - IGNORÉE")
        labels_symlink = dataset_output / "labels"
        label_dir = Path("var_home/datasets/gta_crop/labels")
        
        if not label_dir.exists():
            print(f"  Source labels not found: {label_dir}")
            
        # Create symbolic link
        if not labels_symlink.exists():
            labels_symlink.symlink_to(label_dir)
            print(f"  ✓ Lien symbolique créé: {labels_symlink} -> {label_dir}")
        else:
            print(f"  Lien symbolique existe déjà: {labels_symlink}")
    
        # Count generated images
        n_generated = len(list((dataset_output / "images").glob("*.png")))
        print(f"  ✓ Dataset généré: {n_generated} images")
        print(f"  ✓ Chemin: {dataset_output}")
        
    except Exception as e:
        print(f"  Erreur lors de la génération: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================
    # STEP 4: DISTS & DREAMSIM EVALUATION
    # ========================================
    print_step(4, 5, "Évaluation DISTS et DreamSim")

    dists_score = None
    dreamsim_score = None

    try:
        from mooss.metrics import compute_dists, compute_dreamsim_full_dataset
        
        image_paths = sorted(list(SYNTHETIC_DIR.glob("*.png")))
        # Récupérer les images sources GTA
        gta_source_images = image_paths[:n_generated]
        
        # Calculer DISTS (distance entre images générées et images sources)
        print("\n  → Calcul DISTS...")
        dists_score = compute_dists(dataset_output / "images", gta_source_images)
        print(f"  ✓ DISTS: {dists_score:.4f}")
        
        # Calculer DreamSim (distance entre images générées et images de style)
        print("\n  → Calcul DreamSim...")
        reference_paths = [p for p in list(STYLE_IMAGES.values()) for _ in range(N_IMAGES_TRAIN)]
        
        dreamsim_score = compute_dreamsim_full_dataset(
            dataset_output / "images", reference_paths)

        print(f"  ✓ DreamSim: {dreamsim_score:.4f}")
        
        # Sauvegarder les résultats
        eval_results = {
            "dists": dists_score,
            "dreamsim": dreamsim_score,
            "n_images": n_generated,
            "n_styles": len(STYLE_IMAGES)
        }
        
        with open(dataset_output / "final_evaluation.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"\n  ✓ Résultats sauvegardés dans final_evaluation.json")
        
    except Exception as e:
        print(f"  Erreur lors de l'évaluation: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # STEP 5: DAFORMER CONFIGURATION & TRAINING
    # ========================================
    print_step(5, 6, "Configuration et entraînement DAFormer")
    
    work_dir = None
    training_success = False
    
    try:
        # Update DAFormer config
        config_file = update_daformer_config(dataset_output)
        
        if RUN_TRAINING:
            work_dir = OUTPUT_BASE_DIR / "daformer_results"
            work_dir.mkdir(exist_ok=True)
            
            training_success = run_daformer_training(work_dir)
            
            if training_success:
                print("  ✓ Entraînement terminé")
            else:
                print("  Entraînement échoué")
        else:
            print("  Entraînement ignoré")
            print(f"\n  Pour lancer manuellement:")
            print(f"    cd var_home/DAFormer")
            print(f"    python run_experiments.py --exp 11")
        
    except Exception as e:
        print(f"  Erreur DAFormer: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # STEP 6: CMMD EVALUATION
    # ========================================
    if RUN_CMMD:
        print_step(6, 6, "Évaluation CMMD")
        
        cmmd_score = None
        try:
            cmmd_score = compute_and_save_cmmd(dataset_output, OUTPUT_BASE_DIR, N_IMAGES_TRAIN)
        except Exception as e:
            print(f"  Erreur: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[6/6] Évaluation CMMD - IGNORÉE")
        cmmd_score = None

    # ========================================
    # SUMMARY
    # ========================================
    
    print_header("PIPELINE TERMINÉ")
    print("Résultats:")
    print(f"  - Optimisation GA: {OUTPUT_BASE_DIR}")
    print(f"  - Dataset généré: {dataset_output}")
    if RUN_GENERATION or dataset_output.exists():
        print(f"  - Images: {n_generated}")
    
    if dists_score is not None or dreamsim_score is not None:
        print(f"\n  Métriques finales:")
        if dists_score is not None:
            print(f"    - DISTS (préservation structure): {dists_score:.4f}")
        if dreamsim_score is not None:
            print(f"    - DreamSim (similarité style): {dreamsim_score:.4f}")
    
    if RUN_TRAINING:
        print(f"  - Config DAFormer: {config_file}")
        if training_success:
            print(f"  - Modèle entraîné: {work_dir}")
    
    if RUN_CMMD and cmmd_score:
        print(f"\n  Métrique CMMD: {cmmd_score:.4f}")
    
    print("\nFichiers générés:")
    print("  - optimization_results.json")
    print("  - pareto_front.png")
    print("  - population_evolution.gif")
    print("  - metadata.json")
    print("  - final_evaluation.json  (DISTS + DreamSim)")
    if RUN_CMMD:
        print("  - cmmd_results.json")
    if RUN_TRAINING:
        print("  - eval_results.json (DAFormer/works_dir)")
    
    print("\nPour restaurer la config DAFormer:")
    print(f"  cp {config_file}.backup {config_file}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nErreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)