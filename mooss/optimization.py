import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.erx import EdgeRecombinationCrossover
from pymoo.decomposition.asf import ASF

from .config import (
    N_DISCRETE_VARS,
    POPULATION_SIZE,
    N_GENERATIONS,
    N_OFFSPRINGS,
    OUTPUT_BASE_DIR,
    ENABLED_METRICS
)
from .problems import StyleTransferSequenceProblem
from .visualization import PopulationGifCallback
from .results import save_optimization_results


def run_optimization():
    """Run the genetic algorithm optimization."""
    
    _print_optimization_header()
    
    problem, algorithm = _setup_optimization()
    gif_callback = PopulationGifCallback(
        metric_names=problem.metric_names,
        output_dir=OUTPUT_BASE_DIR
    )
    
    res = minimize(
        problem,
        algorithm,
        ('n_gen', N_GENERATIONS),
        seed=42,
        verbose=True,
        callback=gif_callback
    )
    
    F = res.F
    approx_ideal = F.min(axis=0)
    approx_nadir = F.max(axis=0)
    nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
    
    weights = np.array([0.4, 0.6]) # Structure, Style
    decomp = ASF()
    i = decomp.do(nF, 1/weights).argmin()
    
    save_optimization_results(res, problem, i)
    _print_optimization_footer(gif_callback)


def _print_optimization_header():
    """Print optimization header information."""
    print("=" * 60)
    print("AUGMENTATION OPTIMIZATION WITH GENETIC ALGORITHM")
    print("=" * 60)
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Generations: {N_GENERATIONS}")
    print(f"Discrete variables (sequence): {N_DISCRETE_VARS}")
    print(f"Enabled metrics: {', '.join(ENABLED_METRICS.keys())}")
    print("=" * 60)


def _setup_optimization():
    """Setup optimization problem and algorithm."""
    
    problem = StyleTransferSequenceProblem(
        metrics_config=ENABLED_METRICS
    )
    
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        n_offsprings=N_OFFSPRINGS,
        sampling=PermutationRandomSampling(),
        crossover=EdgeRecombinationCrossover(),
        mutation=InversionMutation(),
        eliminate_duplicates=True
    )
    
    return problem, algorithm


def _print_optimization_footer(gif_callback):
    """Print optimization footer with output paths."""
    print(f"\n{'='*60}")
    print(f"Population evolution GIF saved to: {gif_callback.gif_path}")
    print(f"Individual frames saved to: {gif_callback.frames_dir}")
    print(f"Pareto front visualization saved to: {OUTPUT_BASE_DIR / 'pareto_front.png'}")
    print(f"{'='*60}")