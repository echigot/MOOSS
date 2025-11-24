import numpy as np
import json
from pymoo.core.problem import ElementwiseProblem

from .config import (
    AUGMENTATION_DICT,
    AUGMENTATION_PARAMS,
    N_DISCRETE_VARS,
    POPULATION_SIZE,
    N_IMAGES_EVAL,
    OUTPUT_BASE_DIR,
    REAL_DIR,
    SYNTHETIC_DIR,
)
from .augmentation import generate_augmented_dataset
from .metrics import MetricsEvaluator


class StyleTransferSequenceProblem(ElementwiseProblem):
    """Optimization of augmentation sequence only with fixed hyperparameters."""
    
    def __init__(self, metrics_config: dict):
        self.generation_counter = 0
        self.population_counter = 0
        
        self.metrics_evaluator = MetricsEvaluator(metrics_config)
        self.metric_names = self.metrics_evaluator.get_metric_names()
        
        # Default hyperparameters
        self.default_hyperparams = {}
        for aug_name, params_config in AUGMENTATION_PARAMS.items():
            self.default_hyperparams[aug_name] = {}
            for param_name, bounds in params_config.items():
                self.default_hyperparams[aug_name][param_name] = bounds["default"]
        
        n_var = N_DISCRETE_VARS
        n_obj = len(self.metric_names)
        xl = np.zeros(n_var)
        xu = np.ones(n_var) * (N_DISCRETE_VARS - 1)
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)
    
    def _evaluate(self, X, out, *args, **kwargs):
        sequence = X.astype(int)
        hyperparams = self.default_hyperparams
        
        self._print_evaluation_info(sequence, hyperparams, "Sequence only")
        
        n_images = self._get_n_images()
        
        if sequence[0] == 0:
            print("  No augmentation applied (stop). Assigning worst metric values.")
            metric_results = {metric_name: float('inf') for metric_name in self.metric_names}
            out["F"] = [float('inf')] * len(self.metric_names)
        else:
            metric_results = self._evaluate_with_augmentation(X, sequence, hyperparams, n_images)
            out["F"] = self.metrics_evaluator.get_objective_array(metric_results)
        
        self._save_evaluation_results(sequence, hyperparams, metric_results)
        self._update_counters()
    
    def _print_evaluation_info(self, sequence, hyperparams, mode):
        """Print evaluation information."""
        print(f"\n{'='*60}")
        print(f"Evaluating Generation {self.generation_counter}, Individual {self.population_counter}")
        print(f"Mode: {mode}")
        print(f"Augmentation sequence: {[AUGMENTATION_DICT[int(x)] for x in sequence]}")
        print(f"Hyperparameters: {hyperparams if mode != 'Sequence only' else 'Default values'}")
        print(f"{'='*60}")
    
    def _get_n_images(self):
        """Get number of images for evaluation."""
        return N_IMAGES_EVAL
    
    def _evaluate_with_augmentation(self, X, sequence, hyperparams, n_images):
        """Evaluate augmentation with metrics."""
        # Reconstruct X with default hyperparams
        hyperparams_flat = []
        for aug_name, params_config in AUGMENTATION_PARAMS.items():
            for param_name in params_config.keys():
                hyperparams_flat.append(hyperparams[aug_name][param_name])
        X_full = np.concatenate([sequence, hyperparams_flat])
        
        dataset_path, image_set = generate_augmented_dataset(
            X_full, 
            self.generation_counter, 
            self.population_counter,
            OUTPUT_BASE_DIR,
            n_images=n_images
        )
        
        metric_results = self.metrics_evaluator.evaluate_all(
            generated_dir=dataset_path / "images",
            real_dir=REAL_DIR,
            synthetic_dir=SYNTHETIC_DIR,
            image_set=image_set
        )
        
        return metric_results
    
    
    def _save_evaluation_results(self, sequence, hyperparams, metric_results):
        """Save evaluation results to JSON."""
        results_dict = {
            "generation": self.generation_counter,
            "individual": self.population_counter,
            "augmentation_sequence": [AUGMENTATION_DICT[int(x)] for x in sequence],
            "hyperparameters": hyperparams,
            **metric_results
        }
        
        dataset_path = OUTPUT_BASE_DIR / f"gen_{self.generation_counter}" / f"ind_{self.population_counter}"
        dataset_path.mkdir(parents=True, exist_ok=True)
        with open(dataset_path / "eval.json", 'w') as f:
            json.dump(results_dict, f, indent=2)
    
    def _update_counters(self):
        """Update generation and population counters."""
        self.population_counter += 1
        if self.population_counter >= POPULATION_SIZE:
            self.population_counter = 0
            self.generation_counter += 1