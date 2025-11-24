import json

from .config import (
    AUGMENTATION_DICT,
    AUGMENTATION_PARAMS,
    N_DISCRETE_VARS,
    POPULATION_SIZE,
    N_IMAGES_EVAL,
    ENABLED_METRICS,
    OUTPUT_BASE_DIR,
    N_GENERATIONS,
)
from .visualization import plot_pareto_front

def save_optimization_results(res, problem, optimal_solution_idx=None):
    """Save optimization results to files."""
    metric_names = problem.metric_names
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Number of solutions in Pareto front: {len(res.F)}")
    print(f"\nPareto front ({', '.join([m.upper() for m in metric_names])}):")
    
    pareto_solutions = _extract_pareto_solutions(res, problem, metric_names, optimal_solution_idx)
    _print_pareto_solutions(pareto_solutions, metric_names)
    
    plot_pareto_front(res.F, pareto_solutions, metric_names, OUTPUT_BASE_DIR, optimal_solution_idx)
    
    _save_results_json(pareto_solutions, metric_names, optimal_solution_idx)


def _extract_pareto_solutions(res, problem, metric_names, optimal_solution_idx=None):
    """Extract Pareto solutions from optimization results."""
    pareto_solutions = []
    
    for i, objectives in enumerate(res.F):
        sequence = res.X[i][:N_DISCRETE_VARS].astype(int)
        aug_sequence = [AUGMENTATION_DICT[int(x)] for x in sequence]
        
        hyperparams = problem.default_hyperparams
        
        is_optimal = (i == optimal_solution_idx) if optimal_solution_idx is not None else False
        solution_info = {
            "solution_id": i + 1,
            "augmentation_sequence": aug_sequence,
            "augmentation_indices": sequence.tolist(),
            "hyperparameters": hyperparams,
            "is_optimal": bool(is_optimal),
        }
        
        for metric_name, value in zip(metric_names, objectives):
            solution_info[metric_name.upper()] = float(value)
        
        # Trouver la génération et l'individu correspondants
        _find_generation_info(solution_info, aug_sequence)
        
        pareto_solutions.append(solution_info)
    
    # Filtrer pour garder uniquement les solutions uniques
    pareto_solutions = _filter_unique_solutions(pareto_solutions)
    
    return pareto_solutions


def _get_sequence_before_stop(aug_sequence):
    """Extract augmentation sequence before 'stop'."""
    try:
        stop_index = aug_sequence.index("stop")
        return tuple(aug_sequence[:stop_index])
    except ValueError:
        # Si 'stop' n'est pas trouvé, utiliser toute la séquence
        return tuple(aug_sequence)


def _filter_unique_solutions(pareto_solutions):
    """Filter solutions to keep only unique sequences, prioritizing earlier generations."""
    unique_solutions = {}
    
    for solution in pareto_solutions:
        sequence_key = _get_sequence_before_stop(solution["augmentation_sequence"])
        generation = solution.get("generation", float('inf'))
        is_optimal = solution.get("is_optimal", False)
        
        if sequence_key not in unique_solutions:
            unique_solutions[sequence_key] = solution
        else:
            # Garder la solution avec la génération la plus petite
            # Mais toujours privilégier la solution optimale
            existing_generation = unique_solutions[sequence_key].get("generation", float('inf'))
            existing_optimal = unique_solutions[sequence_key].get("is_optimal", False)
            
            if is_optimal or (not existing_optimal and generation < existing_generation):
                unique_solutions[sequence_key] = solution
    
    # Réattribuer les IDs de solution
    filtered_solutions = list(unique_solutions.values())
    for i, solution in enumerate(filtered_solutions):
        solution["solution_id"] = i + 1
    
    return filtered_solutions


def _find_generation_info(solution_info, aug_sequence):
    """Find generation and individual information for a solution."""
    for ind_dir in sorted(OUTPUT_BASE_DIR.rglob("ind_*")):
        eval_file = ind_dir / "eval.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                eval_data = json.load(f)
                if eval_data.get("augmentation_sequence") == aug_sequence:
                    solution_info["generation"] = eval_data.get("generation")
                    solution_info["individual"] = eval_data.get("individual")
                    solution_info["dataset_path"] = str(ind_dir)
                    break


def _print_pareto_solutions(pareto_solutions, metric_names):
    """Print Pareto solutions to console."""
    for solution in pareto_solutions:
        print(f"\nSolution {solution['solution_id']}:")
        for metric_name in metric_names:
            print(f"  {metric_name.upper()}: {solution[metric_name.upper()]:.4f}")
        print(f"  Augmentation sequence: {solution['augmentation_sequence']}")
        print(f"  Hyperparameters: {solution['hyperparameters']}")


def _save_results_json(pareto_solutions, metric_names, optimal_solution_idx=None):
    """Save results to JSON file."""
    results_path = OUTPUT_BASE_DIR / "optimization_results.json"
    
    # Trouver la solution optimale dans les solutions filtrées
    optimal_solution = None
    if optimal_solution_idx is not None:
        for sol in pareto_solutions:
            if sol.get("is_optimal", False):
                optimal_solution = sol
                break
    
    results = {
        "configuration": {
            "population_size": POPULATION_SIZE,
            "n_generations": N_GENERATIONS,
            "n_images_eval": N_IMAGES_EVAL,
            "n_discrete_vars": N_DISCRETE_VARS,
            "enabled_metrics": list(ENABLED_METRICS.keys()),
            "augmentation_params": AUGMENTATION_PARAMS,
        },
        "pareto_front": {
            "n_solutions": len(pareto_solutions),
            "solutions": pareto_solutions,
        },
        "optimal_solution": optimal_solution,
        "augmentation_dictionary": AUGMENTATION_DICT,
        "metrics_config": ENABLED_METRICS,
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    if optimal_solution:
        print(f"Optimal solution (ID {optimal_solution['solution_id']}) marked in results.")