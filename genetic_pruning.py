"""
Genetic Algorithm for Adaptive Pruning in LLM-Sieve

This module implements the GA-based search for optimal pruning factors
across different matrices in the LLM.
"""

import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
import random
from tqdm import tqdm


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 100
    num_generations: int = 30
    crossover_prob: float = 0.5
    mutation_prob: float = 0.2
    tournament_size: int = 5
    stagnation_threshold: int = 10  # Generations without improvement
    improvement_threshold: float = 0.05  # 5% improvement to continue
    
    # Allowed pruning factors
    allowed_factors: List[float] = None
    
    def __post_init__(self):
        if self.allowed_factors is None:
            self.allowed_factors = [1.0, 0.9, 0.75, 0.6, 0.5, 
                                   0.35, 0.25, 0.2, 0.1, 0.05]


class Chromosome:
    """
    Represents a candidate pruning configuration.
    
    Each gene corresponds to a pruning factor for a specific matrix.
    """
    
    def __init__(self, 
                 pruning_factors: Dict[str, float],
                 fitness: float = 0.0):
        """
        Args:
            pruning_factors: Dictionary mapping matrix names to pruning factors
            fitness: Fitness score (higher is better)
        """
        self.pruning_factors = pruning_factors.copy()
        self.fitness = fitness
        self.accuracy = 0.0
        self.compression = 0.0
        
    def copy(self) -> 'Chromosome':
        """Create a deep copy of this chromosome."""
        return Chromosome(self.pruning_factors.copy(), self.fitness)
    
    def __repr__(self) -> str:
        return f"Chromosome(fitness={self.fitness:.4f}, compression={self.compression:.2%})"


class GeneticPruningOptimizer:
    """
    Genetic Algorithm for finding optimal adaptive pruning configuration.
    
    The GA searches for a pruning factor vector that maximizes compression
    while maintaining accuracy within a tolerance.
    """
    
    def __init__(self,
                 matrix_names: List[str],
                 layer_dims: Dict[str, Tuple[int, int]],
                 evaluate_fn: Callable[[Dict[str, float]], float],
                 target_accuracy: float,
                 config: GeneticConfig = None):
        """
        Args:
            matrix_names: List of matrix names to optimize
            layer_dims: Dictionary mapping matrix names to (H, D) dimensions
            evaluate_fn: Function that takes pruning_factors dict and returns accuracy
            target_accuracy: Minimum acceptable accuracy
            config: GA configuration
        """
        self.matrix_names = matrix_names
        self.layer_dims = layer_dims
        self.evaluate_fn = evaluate_fn
        self.target_accuracy = target_accuracy
        self.config = config or GeneticConfig()
        
        self.population: List[Chromosome] = []
        self.best_chromosome: Chromosome = None
        self.generation_history: List[Dict] = []
        
    def _compute_compression(self, pruning_factors: Dict[str, float]) -> float:
        """Compute compression ratio for a given configuration."""
        total_original = 0
        total_pruned = 0
        
        for name, (H, D) in self.layer_dims.items():
            if name not in pruning_factors:
                continue
                
            p = pruning_factors[name]
            R = int(p * H * D / (H + D))
            R = max(1, min(R, min(H, D)))
            
            total_original += H * D
            total_pruned += R * (H + D)
        
        return 1.0 - (total_pruned / total_original)
    
    def _compute_fitness(self, 
                        accuracy: float, 
                        compression: float,
                        a0: float) -> float:
        """
        Compute fitness with thresholded exponential penalty.
        
        Fitness = compression × (1 + exp^(50*(accuracy - a0)))
        
        Configurations above threshold a0 are rewarded,
        underperforming ones are penalized exponentially.
        """
        penalty_factor = 1.0 + np.exp(50.0 * (accuracy - a0))
        return compression * penalty_factor
    
    def _initialize_population(self) -> None:
        """Initialize population with random configurations."""
        self.population = []
        
        # Add some uniform pruning configurations
        num_uniform = min(10, self.config.population_size // 10)
        for _ in range(num_uniform):
            factor = random.choice(self.config.allowed_factors)
            pruning_factors = {name: factor for name in self.matrix_names}
            self.population.append(Chromosome(pruning_factors))
        
        # Fill rest with random configurations
        while len(self.population) < self.config.population_size:
            pruning_factors = {
                name: random.choice(self.config.allowed_factors)
                for name in self.matrix_names
            }
            self.population.append(Chromosome(pruning_factors))
    
    def _evaluate_population(self, verbose: bool = False) -> None:
        """Evaluate fitness for all chromosomes in population."""
        if verbose:
            print("Evaluating population...")
            pbar = tqdm(self.population, desc="Evaluating chromosomes")
        else:
            pbar = self.population
            
        for chromosome in pbar:
            # Evaluate accuracy
            accuracy = self.evaluate_fn(chromosome.pruning_factors)
            compression = self._compute_compression(chromosome.pruning_factors)
            
            # Compute fitness
            fitness = self._compute_fitness(accuracy, compression, self.target_accuracy)
            
            chromosome.accuracy = accuracy
            chromosome.compression = compression
            chromosome.fitness = fitness
    
    def _tournament_selection(self) -> Chromosome:
        """Select a chromosome using tournament selection."""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda c: c.fitness)
    
    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """
        Perform crossover between two parents.
        
        Split at random point and combine.
        """
        if random.random() > self.config.crossover_prob:
            return parent1.copy()
        
        # Get matrix names in consistent order
        names = list(parent1.pruning_factors.keys())
        split_point = random.randint(1, len(names) - 1)
        
        # Create child by combining parents
        child_factors = {}
        for i, name in enumerate(names):
            if i < split_point:
                child_factors[name] = parent1.pruning_factors[name]
            else:
                child_factors[name] = parent2.pruning_factors[name]
        
        return Chromosome(child_factors)
    
    def _mutate(self, chromosome: Chromosome) -> None:
        """
        Mutate a chromosome by perturbing pruning factors.
        
        With mutation probability, change a factor by one step
        in the allowed factors list.
        """
        for name in chromosome.pruning_factors:
            if random.random() < self.config.mutation_prob:
                current_factor = chromosome.pruning_factors[name]
                
                # Find neighboring factors
                try:
                    current_idx = self.config.allowed_factors.index(current_factor)
                except ValueError:
                    # If not in list, pick random
                    chromosome.pruning_factors[name] = random.choice(
                        self.config.allowed_factors
                    )
                    continue
                
                # Move one step up or down
                if current_idx == 0:
                    new_idx = 1
                elif current_idx == len(self.config.allowed_factors) - 1:
                    new_idx = current_idx - 1
                else:
                    new_idx = current_idx + random.choice([-1, 1])
                
                chromosome.pruning_factors[name] = self.config.allowed_factors[new_idx]
    
    def _evolve_generation(self) -> None:
        """Create next generation through selection, crossover, and mutation."""
        new_population = []
        
        # Elitism: keep best chromosome
        best = max(self.population, key=lambda c: c.fitness)
        new_population.append(best.copy())
        
        # Generate rest of population
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
    
    def optimize(self, verbose: bool = True) -> Chromosome:
        """
        Run genetic algorithm to find optimal pruning configuration.
        
        Args:
            verbose: Print progress information
            
        Returns:
            Best chromosome found
        """
        if verbose:
            print(f"Starting Genetic Algorithm Optimization")
            print(f"Population size: {self.config.population_size}")
            print(f"Max generations: {self.config.num_generations}")
            print(f"Target accuracy: {self.target_accuracy:.2%}\n")
        
        # Initialize population
        self._initialize_population()
        
        # Track best fitness for stagnation detection
        best_fitness_history = []
        stagnation_count = 0
        
        for generation in range(self.config.num_generations):
            # Evaluate current population
            self._evaluate_population(verbose=verbose and generation == 0)
            
            # Track best chromosome
            current_best = max(self.population, key=lambda c: c.fitness)
            if self.best_chromosome is None or current_best.fitness > self.best_chromosome.fitness:
                self.best_chromosome = current_best.copy()
            
            # Record generation statistics
            fitnesses = [c.fitness for c in self.population]
            accuracies = [c.accuracy for c in self.population]
            compressions = [c.compression for c in self.population]
            
            gen_stats = {
                'generation': generation,
                'best_fitness': max(fitnesses),
                'avg_fitness': np.mean(fitnesses),
                'best_accuracy': max(accuracies),
                'best_compression': max(compressions),
                'best_chromosome': current_best.copy()
            }
            self.generation_history.append(gen_stats)
            
            if verbose:
                print(f"Generation {generation + 1}/{self.config.num_generations}")
                print(f"  Best fitness: {gen_stats['best_fitness']:.4f}")
                print(f"  Best accuracy: {gen_stats['best_accuracy']:.2%}")
                print(f"  Best compression: {gen_stats['best_compression']:.2%}")
                print(f"  Avg fitness: {gen_stats['avg_fitness']:.4f}\n")
            
            # Check for stagnation
            best_fitness_history.append(gen_stats['best_fitness'])
            if len(best_fitness_history) >= self.config.stagnation_threshold:
                recent_best = max(best_fitness_history[-self.config.stagnation_threshold:])
                older_best = max(best_fitness_history[:-self.config.stagnation_threshold])
                improvement = (recent_best - older_best) / (older_best + 1e-10)
                
                if improvement < self.config.improvement_threshold:
                    stagnation_count += 1
                    if stagnation_count >= self.config.stagnation_threshold:
                        if verbose:
                            print(f"Stopping: No significant improvement for "
                                  f"{self.config.stagnation_threshold} generations")
                        break
                else:
                    stagnation_count = 0
            
            # Evolve to next generation
            self._evolve_generation()
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Best configuration:")
            print(f"  Accuracy: {self.best_chromosome.accuracy:.2%}")
            print(f"  Compression: {self.best_chromosome.compression:.2%}")
            print(f"  Fitness: {self.best_chromosome.fitness:.4f}")
        
        return self.best_chromosome
    
    def get_pruning_statistics(self) -> Dict:
        """Get statistics about the final pruning configuration."""
        if self.best_chromosome is None:
            return {}
        
        # Analyze pruning factors by matrix type
        stats = {
            'overall_compression': self.best_chromosome.compression,
            'final_accuracy': self.best_chromosome.accuracy,
            'pruning_factors': self.best_chromosome.pruning_factors.copy()
        }
        
        # Group by matrix type (e.g., all wqkv, all w1, etc.)
        matrix_type_factors = {}
        for name, factor in self.best_chromosome.pruning_factors.items():
            # Extract matrix type (last part after underscore)
            matrix_type = name.split('_')[-1]
            if matrix_type not in matrix_type_factors:
                matrix_type_factors[matrix_type] = []
            matrix_type_factors[matrix_type].append(factor)
        
        # Compute averages
        stats['avg_by_type'] = {
            mtype: np.mean(factors)
            for mtype, factors in matrix_type_factors.items()
        }
        
        return stats


# Example usage
if __name__ == "__main__":
    print("Genetic Algorithm for Adaptive Pruning")
    print("=" * 50)
    
    # Dummy example
    matrix_names = [f"layer{i}_{mtype}" 
                   for i in range(4) 
                   for mtype in ['wqkv', 'wo', 'w1', 'w2']]
    
    layer_dims = {name: (1024, 512) for name in matrix_names}
    
    def dummy_evaluate(pruning_factors):
        """Dummy evaluation function."""
        compression = np.mean(list(pruning_factors.values()))
        # Simulate accuracy drop with more compression
        accuracy = 0.95 - (1.0 - compression) * 0.3
        return accuracy
    
    config = GeneticConfig(population_size=20, num_generations=10)
    optimizer = GeneticPruningOptimizer(
        matrix_names=matrix_names,
        layer_dims=layer_dims,
        evaluate_fn=dummy_evaluate,
        target_accuracy=0.90,
        config=config
    )
    
    best = optimizer.optimize(verbose=True)
    
    print("\n" + "=" * 50)
    print("Final Statistics:")
    stats = optimizer.get_pruning_statistics()
    print(f"Overall compression: {stats['overall_compression']:.2%}")
    print(f"Final accuracy: {stats['final_accuracy']:.2%}")
    print("\nAverage pruning by matrix type:")
    for mtype, avg_factor in stats['avg_by_type'].items():
        print(f"  {mtype}: {avg_factor:.3f}")
