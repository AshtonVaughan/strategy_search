"""
Evolutionary algorithm for searching optimal trading strategies.
"""

import random
import copy
import numpy as np
from typing import List, Dict, Tuple


class EvolutionarySearcher:
    """
    Genetic algorithm for finding optimal trading strategy configurations.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Search configuration
        """
        self.config = config
        self.population_size = config['search']['population_size']
        self.mutation_rate = config['evolution']['mutation_rate']
        self.crossover_rate = config['evolution']['crossover_rate']
        self.survival_rate = config['evolution']['survival_rate']

        # Search spaces
        self.model_space = config['model']
        self.training_space = config['training']
        self.strategy_space = config['strategy']

    def initialize_population(self) -> List[Dict]:
        """
        Create initial population of random configurations.

        Returns:
            List of random strategy configurations
        """
        population = []

        for _ in range(self.population_size):
            config = self._random_config()
            population.append(config)

        return population

    def _random_config(self) -> Dict:
        """Generate a single random configuration."""
        return {
            # Model architecture
            'hidden_size': random.choice(self.model_space['hidden_size']),
            'num_layers': random.choice(self.model_space['num_layers']),
            'num_heads': random.choice(self.model_space['num_heads']),
            'seq_length': random.choice(self.model_space['seq_length']),
            'dropout': random.choice(self.model_space['dropout']),

            # Training hyperparameters
            'learning_rate': random.choice(self.training_space['learning_rate']),
            'batch_size': random.choice(self.training_space['batch_size']),
            'epochs': random.choice(self.training_space['epochs']),
            'weight_decay': random.choice(self.training_space['weight_decay']),

            # Trading strategy
            'confidence_threshold': random.choice(self.strategy_space['confidence_threshold']),
            'position_sizing': random.choice(self.strategy_space['position_sizing']),
            'stop_loss_pips': random.choice(self.strategy_space['stop_loss_pips']),
            'take_profit_pips': random.choice(self.strategy_space['take_profit_pips']),
            'trailing_stop': random.choice(self.strategy_space['trailing_stop']),
            'risk_per_trade': random.choice(self.strategy_space['risk_per_trade']),
        }

    def evolve_generation(
        self,
        population: List[Dict],
        fitnesses: List[float]
    ) -> List[Dict]:
        """
        Evolve population to create next generation.

        Args:
            population: Current population
            fitnesses: Fitness scores for each individual

        Returns:
            New population
        """
        # Sort by fitness (descending)
        sorted_pop = [x for _, x in sorted(zip(fitnesses, population),
                                           key=lambda pair: pair[0],
                                           reverse=True)]

        new_population = []

        # 1. Elitism: Keep top performers unchanged
        n_elite = int(self.population_size * self.survival_rate)
        new_population.extend(sorted_pop[:n_elite])

        # 2. Crossover: Combine top performers
        n_crossover = int(self.population_size * self.crossover_rate)
        for _ in range(n_crossover):
            parent1 = random.choice(sorted_pop[:n_elite * 2])
            parent2 = random.choice(sorted_pop[:n_elite * 2])
            child = self._crossover(parent1, parent2)
            new_population.append(child)

        # 3. Mutation: Mutate top performers
        n_mutants = int(self.population_size * self.mutation_rate)
        for _ in range(n_mutants):
            parent = random.choice(sorted_pop[:n_elite * 2])
            mutant = self._mutate(parent)
            new_population.append(mutant)

        # 4. Random exploration: Completely new configs
        n_random = self.population_size - len(new_population)
        for _ in range(n_random):
            new_population.append(self._random_config())

        return new_population

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        Create child by combining traits from two parents.

        Args:
            parent1: First parent configuration
            parent2: Second parent configuration

        Returns:
            Child configuration
        """
        child = {}

        for key in parent1.keys():
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]

        return child

    def _mutate(self, config: Dict) -> Dict:
        """
        Mutate a configuration by changing 1-3 random parameters.

        Args:
            config: Configuration to mutate

        Returns:
            Mutated configuration
        """
        mutated = copy.deepcopy(config)

        # Randomly change 1-3 parameters
        num_mutations = random.randint(1, 3)
        params_to_mutate = random.sample(list(mutated.keys()), num_mutations)

        for param in params_to_mutate:
            mutated[param] = self._get_random_value(param)

        return mutated

    def _get_random_value(self, param: str):
        """Get a random value for a parameter from search space."""
        # Model params
        if param == 'hidden_size':
            return random.choice(self.model_space['hidden_size'])
        elif param == 'num_layers':
            return random.choice(self.model_space['num_layers'])
        elif param == 'num_heads':
            return random.choice(self.model_space['num_heads'])
        elif param == 'seq_length':
            return random.choice(self.model_space['seq_length'])
        elif param == 'dropout':
            return random.choice(self.model_space['dropout'])

        # Training params
        elif param == 'learning_rate':
            return random.choice(self.training_space['learning_rate'])
        elif param == 'batch_size':
            return random.choice(self.training_space['batch_size'])
        elif param == 'epochs':
            return random.choice(self.training_space['epochs'])
        elif param == 'weight_decay':
            return random.choice(self.training_space['weight_decay'])

        # Strategy params
        elif param == 'confidence_threshold':
            return random.choice(self.strategy_space['confidence_threshold'])
        elif param == 'position_sizing':
            return random.choice(self.strategy_space['position_sizing'])
        elif param == 'stop_loss_pips':
            return random.choice(self.strategy_space['stop_loss_pips'])
        elif param == 'take_profit_pips':
            return random.choice(self.strategy_space['take_profit_pips'])
        elif param == 'trailing_stop':
            return random.choice(self.strategy_space['trailing_stop'])
        elif param == 'risk_per_trade':
            return random.choice(self.strategy_space['risk_per_trade'])

        else:
            return None


def calculate_diversity_score(population: List[Dict]) -> float:
    """
    Calculate diversity of current population.
    Higher diversity = more exploration.

    Args:
        population: List of configurations

    Returns:
        Diversity score (0-1)
    """
    if len(population) < 2:
        return 0.0

    # Calculate pairwise differences
    total_diff = 0
    comparisons = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            diff = sum(population[i][k] != population[j][k] for k in population[i].keys())
            total_diff += diff
            comparisons += 1

    # Average difference as a fraction of total parameters
    avg_diff = total_diff / comparisons if comparisons > 0 else 0
    diversity = avg_diff / len(population[0])  # Normalize by number of parameters

    return diversity
