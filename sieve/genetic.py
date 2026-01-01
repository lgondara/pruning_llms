"""
Genetic Algorithm Optimizer for Adaptive Pruning

Finds optimal per-layer pruning factors that maximize compression
while maintaining accuracy above threshold.
"""

import numpy as np
from typing import Callable, List, Tuple
from tqdm import tqdm


class GeneticOptimizer:
    """
    Genetic algorithm for finding optimal pruning factors.
    
    Each individual is a vector of pruning factors, one per layer.
    Fitness balances compression ratio vs accuracy preservation.
    """
    
    def __init__(
        self,
        n_genes: int,
        population_size: int = 100,
        generations: int = 30,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.2,
        gene_range: Tuple[float, float] = (0.2, 0.9),
        elite_ratio: float = 0.1
    ):
        """
        Args:
            n_genes: Number of layers (genes per individual)
            population_size: Number of individuals
            generations: Number of evolution iterations
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation per gene
            gene_range: (min, max) range for pruning factors
            elite_ratio: Fraction of top individuals to preserve
        """
        self.n_genes = n_genes
        self.pop_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.gene_min, self.gene_max = gene_range
        self.n_elite = max(1, int(elite_ratio * population_size))
    
    def initialize_population(self) -> np.ndarray:
        """Create initial random population."""
        return np.random.uniform(
            self.gene_min, self.gene_max,
            (self.pop_size, self.n_genes)
        )
    
    def evaluate_population(
        self,
        population: np.ndarray,
        fitness_fn: Callable[[List[float]], float]
    ) -> np.ndarray:
        """Evaluate fitness of all individuals."""
        fitness = np.zeros(self.pop_size)
        
        for i in tqdm(range(self.pop_size), desc="Evaluating", leave=False):
            fitness[i] = fitness_fn(population[i].tolist())
        
        return fitness
    
    def select_parents(
        self,
        population: np.ndarray,
        fitness: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select two parents using tournament selection."""
        def tournament(k: int = 3) -> int:
            candidates = np.random.choice(self.pop_size, k, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            return winner
        
        parent1 = population[tournament()]
        parent2 = population[tournament()]
        return parent1, parent2
    
    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform uniform crossover."""
        if np.random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        # Uniform crossover: each gene from either parent
        mask = np.random.random(self.n_genes) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        return child1, child2
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """Apply Gaussian mutation."""
        mutant = individual.copy()
        
        for i in range(self.n_genes):
            if np.random.random() < self.mutation_prob:
                # Gaussian perturbation
                mutant[i] += np.random.normal(0, 0.1)
                # Clip to valid range
                mutant[i] = np.clip(mutant[i], self.gene_min, self.gene_max)
        
        return mutant
    
    def evolve(
        self,
        population: np.ndarray,
        fitness: np.ndarray
    ) -> np.ndarray:
        """Create next generation."""
        new_population = np.zeros_like(population)
        
        # Elitism: keep best individuals
        elite_idx = np.argsort(fitness)[-self.n_elite:]
        new_population[:self.n_elite] = population[elite_idx]
        
        # Fill rest with offspring
        idx = self.n_elite
        while idx < self.pop_size:
            parent1, parent2 = self.select_parents(population, fitness)
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population[idx] = child1
            idx += 1
            if idx < self.pop_size:
                new_population[idx] = child2
                idx += 1
        
        return new_population
    
    def optimize(
        self,
        fitness_fn: Callable[[List[float]], float],
        verbose: bool = True
    ) -> List[float]:
        """
        Run genetic algorithm optimization.
        
        Args:
            fitness_fn: Function that takes pruning factors and returns fitness
            verbose: Print progress
        
        Returns:
            Best pruning factors found
        """
        if verbose:
            print(f"\nGenetic Algorithm Optimization")
            print(f"Population: {self.pop_size}, Generations: {self.generations}")
            print(f"Genes (layers): {self.n_genes}")
        
        # Initialize
        population = self.initialize_population()
        best_fitness = -np.inf
        best_individual = None
        
        for gen in range(self.generations):
            # Evaluate
            fitness = self.evaluate_population(population, fitness_fn)
            
            # Track best
            gen_best_idx = np.argmax(fitness)
            gen_best_fitness = fitness[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()
            
            if verbose:
                avg_factor = np.mean(population[gen_best_idx])
                print(f"Gen {gen+1:3d}: Best fitness={gen_best_fitness:.4f}, "
                      f"Avg factor={avg_factor:.3f}, "
                      f"Compression≈{1-avg_factor:.1%}")
            
            # Evolve (skip on last generation)
            if gen < self.generations - 1:
                population = self.evolve(population, fitness)
        
        if verbose:
            print(f"\nOptimization complete!")
            print(f"Best fitness: {best_fitness:.4f}")
            avg = np.mean(best_individual)
            print(f"Average pruning factor: {avg:.3f} ({1-avg:.1%} compression)")
        
        return best_individual.tolist()


class AdaptivePruningSearch:
    """
    Alternative: Binary search per layer type for simpler adaptive pruning.
    
    Less optimal than GA but much faster.
    """
    
    def __init__(
        self,
        layer_types: List[str],
        search_range: Tuple[float, float] = (0.3, 0.8),
        search_steps: int = 5
    ):
        self.layer_types = list(set(layer_types))  # Unique types
        self.search_range = search_range
        self.search_steps = search_steps
    
    def search(
        self,
        eval_fn: Callable[[dict], float],
        threshold: float,
        layer_type_map: dict
    ) -> dict:
        """
        Search for optimal factor per layer type.
        
        Args:
            eval_fn: Takes {layer_name: factor} dict, returns accuracy
            threshold: Minimum accuracy
            layer_type_map: {layer_name: layer_type}
        
        Returns:
            Optimal {layer_name: factor} dict
        """
        type_factors = {}
        
        for ltype in self.layer_types:
            print(f"\nSearching optimal factor for {ltype} layers...")
            
            low, high = self.search_range
            best_factor = high  # Conservative default
            
            # Binary search
            for _ in range(self.search_steps):
                mid = (low + high) / 2
                
                # Create factor dict with current mid for this type
                factors = {}
                for name, t in layer_type_map.items():
                    if t == ltype:
                        factors[name] = mid
                    else:
                        factors[name] = type_factors.get(t, 0.5)
                
                accuracy = eval_fn(factors)
                
                if accuracy >= threshold:
                    best_factor = mid
                    high = mid  # Can try more aggressive
                else:
                    low = mid  # Need more conservative
            
            type_factors[ltype] = best_factor
            print(f"  {ltype}: {best_factor:.3f} ({1-best_factor:.1%} compression)")
        
        # Build final factor dict
        final_factors = {}
        for name, ltype in layer_type_map.items():
            final_factors[name] = type_factors[ltype]
        
        return final_factors
