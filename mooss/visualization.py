import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from pymoo.core.callback import Callback
import matplotlib.cm as cm


class PopulationGifCallback(Callback):
    """Callback to create and update GIF of population evolution."""
    
    def __init__(self, metric_names, output_dir):
        super().__init__()
        self.metric_names = metric_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer un dossier pour les frames PNG
        self.frames_dir = self.output_dir / "population_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        self.generations_data = []
        self.gif_path = self.output_dir / "population_evolution.gif"
        
        # Limites fixes entre 0 et 1 pour toutes les métriques
        self.fixed_limits = {'min': 0.0, 'max': 1.0}
        
        # Préparer les paires de métriques à visualiser
        if len(metric_names) >= 2:
            self.metric_pairs = [(0, 1)]
            if len(metric_names) > 2:
                for i in range(len(metric_names)):
                    for j in range(i + 1, len(metric_names)):
                        if (i, j) not in self.metric_pairs:
                            self.metric_pairs.append((i, j))
        else:
            self.metric_pairs = []
    
    def notify(self, algorithm):
        """Called after each generation."""
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")
        
        if F is not None and len(F) > 0:
            self.generations_data.append({
                "generation": gen,
                "objectives": F.copy()
            })
            self._save_current_frame(gen)
            self._update_gif()
    
    def _save_current_frame(self, gen):
        """Save current generation as PNG frame."""
        if len(self.metric_pairs) == 0 or len(self.generations_data) == 0:
            return
        
        n_pairs = len(self.metric_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
        
        if n_pairs == 1:
            axes = [axes]
        
        gen_data = self.generations_data[-1]
        objectives = gen_data["objectives"]
        valid_mask = ~np.any(np.isinf(objectives), axis=1)
        objectives_valid = objectives[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        for idx, (i, j) in enumerate(self.metric_pairs):
            ax = axes[idx]
            
            if len(objectives_valid) > 0:
                ax.scatter(objectives_valid[:, i], objectives_valid[:, j], 
                         c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1.5,
                         label=f'Gen {gen}')
                
                # Annoter chaque individu avec son numéro
                for ind_idx, (x, y) in enumerate(zip(objectives_valid[:, i], objectives_valid[:, j])):
                    original_idx = valid_indices[ind_idx]
                    ax.annotate(
                        f'{original_idx + 1}',
                        (x, y),
                        ha='center',
                        va='center',
                        fontsize=8,
                        fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle='circle,pad=0.1', facecolor='blue', alpha=0.8, edgecolor='none')
                    )
                
                # Afficher les générations précédentes
                for prev_idx in range(max(0, len(self.generations_data) - 3), len(self.generations_data) - 1):
                    prev_data = self.generations_data[prev_idx]
                    prev_objectives = prev_data["objectives"]
                    prev_valid_mask = ~np.any(np.isinf(prev_objectives), axis=1)
                    prev_objectives_valid = prev_objectives[prev_valid_mask]
                    
                    if len(prev_objectives_valid) > 0:
                        alpha = 0.2 * (prev_idx - max(0, len(self.generations_data) - 3) + 1) / 3
                        ax.scatter(prev_objectives_valid[:, i], prev_objectives_valid[:, j],
                                 c='gray', s=50, alpha=alpha, edgecolors='none')
            
            self._configure_axis(ax, i, j, gen)
        
        plt.tight_layout()
        frame_path = self.frames_dir / f"gen_{gen:04d}.png"
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Frame saved: {frame_path}")
    
    def _configure_axis(self, ax, i, j, gen):
        """Configure axis labels, title, and limits."""
        ax.set_xlabel(f'{self.metric_names[i].upper()}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{self.metric_names[j].upper()}', fontsize=12, fontweight='bold')
        ax.set_title(f'{self.metric_names[i].upper()} vs {self.metric_names[j].upper()}\nGeneration {gen}',
                   fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')
        ax.set_xlim(self.fixed_limits['min'], self.fixed_limits['max'])
        ax.set_ylim(self.fixed_limits['min'], self.fixed_limits['max'])
    
    def _update_gif(self):
        """Update the GIF with all generations so far."""
        if len(self.metric_pairs) == 0 or len(self.generations_data) == 0:
            return
        
        n_pairs = len(self.metric_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
        
        if n_pairs == 1:
            axes = [axes]
        
        def animate(frame_idx):
            for ax in axes:
                ax.clear()
            
            gen_data = self.generations_data[frame_idx]
            gen = gen_data["generation"]
            objectives = gen_data["objectives"]
            valid_mask = ~np.any(np.isinf(objectives), axis=1)
            objectives_valid = objectives[valid_mask]
            valid_indices = np.where(valid_mask)[0]
            
            for idx, (i, j) in enumerate(self.metric_pairs):
                ax = axes[idx]
                
                if len(objectives_valid) > 0:
                    ax.scatter(objectives_valid[:, i], objectives_valid[:, j], 
                             c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1.5,
                             label=f'Gen {gen}')
                    
                    # Annoter chaque individu avec son numéro
                    for ind_idx, (x, y) in enumerate(zip(objectives_valid[:, i], objectives_valid[:, j])):
                        original_idx = valid_indices[ind_idx]
                        ax.annotate(
                            f'{original_idx + 1}',
                            (x, y),
                            ha='center',
                            va='center',
                            fontsize=8,
                            fontweight='bold',
                            color='white',
                            bbox=dict(boxstyle='circle,pad=0.1', facecolor='blue', alpha=0.8, edgecolor='none')
                        )
                    
                    for prev_idx in range(max(0, frame_idx - 2), frame_idx):
                        prev_data = self.generations_data[prev_idx]
                        prev_objectives = prev_data["objectives"]
                        prev_valid_mask = ~np.any(np.isinf(prev_objectives), axis=1)
                        prev_objectives_valid = prev_objectives[prev_valid_mask]
                        
                        if len(prev_objectives_valid) > 0:
                            alpha = 0.2 * (prev_idx - max(0, frame_idx - 2) + 1) / 3
                            ax.scatter(prev_objectives_valid[:, i], prev_objectives_valid[:, j],
                                     c='gray', s=50, alpha=alpha, edgecolors='none')
                
                self._configure_axis(ax, i, j, gen)
            
            plt.tight_layout()
        
        anim = FuncAnimation(fig, animate, frames=len(self.generations_data), 
                           interval=500, repeat=True)
        writer = PillowWriter(fps=2)
        anim.save(self.gif_path, writer=writer)
        plt.close(fig)
        print(f"Updated GIF saved to: {self.gif_path}")


def plot_pareto_front(objectives, pareto_solutions, metric_names, output_dir, optimal_solution_idx=None):
    """Plot Pareto front with generation information and optimal solution highlighted."""
    output_dir = Path(output_dir)
    
    # S'assurer que les tableaux ont la même longueur
    n_solutions = len(pareto_solutions)
    if len(objectives) != n_solutions:
        print(f"Warning: objectives length ({len(objectives)}) != pareto_solutions length ({n_solutions})")
        n_solutions = min(len(objectives), n_solutions)
        objectives = objectives[:n_solutions]
        pareto_solutions = pareto_solutions[:n_solutions]
    
    generations = np.array([sol.get("generation", 0) for sol in pareto_solutions])
    is_optimal = np.array([sol.get("is_optimal", False) for sol in pareto_solutions])
    
    # Créer le masque de validité
    valid_mask = ~np.any(np.isinf(objectives), axis=1)
    
    # Appliquer le masque à tous les tableaux
    objectives_valid = objectives[valid_mask]
    generations_valid = generations[valid_mask]
    is_optimal_valid = is_optimal[valid_mask]
    
    if len(objectives_valid) == 0:
        print("No valid solutions to plot in Pareto front.")
        return
    
    n_metrics = len(metric_names)
    if n_metrics < 2:
        print("Need at least 2 metrics to plot Pareto front.")
        return
    
    metric_pairs = [(i, j) for i in range(n_metrics) for j in range(i + 1, n_metrics)]
    n_pairs = len(metric_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs, 6))
    
    if n_pairs == 1:
        axes = [axes]
    
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(vmin=generations_valid.min(), vmax=generations_valid.max())
    
    for idx, (i, j) in enumerate(metric_pairs):
        ax = axes[idx]
        
        # Séparer les solutions normales et la solution optimale
        normal_mask = ~is_optimal_valid
        optimal_mask = is_optimal_valid
        
        # Tracer les solutions normales
        if np.any(normal_mask):
            scatter = ax.scatter(
                objectives_valid[normal_mask, i], 
                objectives_valid[normal_mask, j],
                c=generations_valid[normal_mask],
                cmap=cmap,
                s=150,
                alpha=0.7,
                edgecolors='black',
                linewidth=2,
                norm=norm,
                label='Pareto solutions'
            )
            
            # Annoter les solutions normales
            for x, y, gen in zip(objectives_valid[normal_mask, i], 
                                objectives_valid[normal_mask, j], 
                                generations_valid[normal_mask]):
                ax.annotate(
                    f'{int(gen)}',
                    (x, y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray')
                )
        
        # Tracer la solution optimale en rouge
        if np.any(optimal_mask):
            ax.scatter(
                objectives_valid[optimal_mask, i], 
                objectives_valid[optimal_mask, j],
                c='red',
                s=250,
                alpha=0.9,
                edgecolors='darkred',
                # linewidth=3,
                # marker='*',
                label='Optimal solution',
                zorder=10
            )
            
            # Annoter la solution optimale
            for x, y, gen in zip(objectives_valid[optimal_mask, i], 
                                objectives_valid[optimal_mask, j], 
                                generations_valid[optimal_mask]):
                ax.annotate(
                    f'OPTIMAL\nGen {int(gen)}',
                    (x, y),
                    xytext=(15, 15),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='darkred',
                    bbox=dict(boxstyle='round,pad=0.5', 
                            #   facecolor='yellow', 
                              alpha=0.9, 
                            #   edgecolor='darkred', 
                            #   linewidth=2
                              )
                )
        
        ax.set_xlabel(f'{metric_names[i].upper()}', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{metric_names[j].upper()}', fontsize=13, fontweight='bold')
        ax.set_title(
            f'Pareto Front: {metric_names[i].upper()} vs {metric_names[j].upper()}',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')
        
        if np.any(normal_mask):
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Generation', fontsize=11, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
        
        x_margin = (objectives_valid[:, i].max() - objectives_valid[:, i].min()) * 0.1
        y_margin = (objectives_valid[:, j].max() - objectives_valid[:, j].min()) * 0.1
        ax.set_xlim(objectives_valid[:, i].min() - x_margin, objectives_valid[:, i].max() + x_margin)
        ax.set_ylim(objectives_valid[:, j].min() - y_margin, objectives_valid[:, j].max() + y_margin)
    
    plt.tight_layout()
    pareto_path = output_dir / "pareto_front.png"
    plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Pareto front plot saved to: {pareto_path}")