"""
Main orchestrator for ML strategy search system.
Searches 1M+ trading strategies over 5-7 days using evolutionary algorithm.
"""

import yaml
import pandas as pd
import numpy as np
import yfinance as yf
import torch
import os
import json
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm

# Import modules
from search.evolution import EvolutionarySearcher
from training.trainer import StrategyTrainer, normalize_data
from training.model import create_model
from backtesting.engine import MLBacktester, calculate_composite_fitness
from validation.cross_validator import CrossValidator, get_training_data_for_week


class StrategySearchOrchestrator:
    """
    Main coordinator for the strategy search system.
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.searcher = EvolutionarySearcher(self.config)
        self.validator = CrossValidator(self.config['validation'])
        self.backtester = MLBacktester(initial_capital=10000)

        # Load data
        print("Loading market data...")
        self.data = self._load_data()

        # Results tracking
        self.all_results = []
        self.best_strategy = None
        self.best_fitness = 0

    def _load_data(self) -> pd.DataFrame:
        """Load EUR/USD data from Yahoo Finance."""
        symbol = self.config['data']['symbol']
        start = self.config['data']['start_date']
        end = self.config['data']['end_date']

        print(f"Fetching {symbol} from {start} to {end}...")
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start, end=end, interval='1h')

        if data.empty:
            raise ValueError("Failed to load data!")

        print(f"Loaded {len(data)} data points")
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]

    def run_search(self):
        """
        Run the complete strategy search.
        """
        print("\n" + "="*70)
        print("STRATEGY SEARCH SYSTEM - STARTING")
        print("="*70)

        # Phase 1: Evolutionary Search
        print("\nPhase 1: Evolutionary Search")
        print(f"Generations: {self.config['search']['generations']}")
        print(f"Population: {self.config['search']['population_size']}")

        population = self.searcher.initialize_population()

        for generation in range(self.config['search']['generations']):
            print(f"\n{'='*70}")
            print(f"Generation {generation + 1}/{self.config['search']['generations']}")
            print(f"{'='*70}")

            # Evaluate all strategies in this generation
            fitnesses = []
            for i, strategy_config in enumerate(population):
                print(f"\nEvaluating strategy {i+1}/{len(population)}...")
                fitness, metrics = self._evaluate_strategy(strategy_config)
                fitnesses.append(fitness)

                # Track results
                result = {
                    'generation': generation + 1,
                    'strategy_id': len(self.all_results),
                    'config': strategy_config,
                    'fitness': fitness,
                    'metrics': metrics
                }
                self.all_results.append(result)

                # Update best
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_strategy = strategy_config
                    print(f"  *** NEW BEST! Fitness: {fitness:.4f}")

            # Print generation summary
            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Best Fitness: {max(fitnesses):.4f}")
            print(f"  Mean Fitness: {np.mean(fitnesses):.4f}")
            print(f"  All-time Best: {self.best_fitness:.4f}")

            # Evolve to next generation
            if generation < self.config['search']['generations'] - 1:
                population = self.searcher.evolve_generation(population, fitnesses)

            # Checkpoint
            if (generation + 1) % self.config['resources']['checkpoint_every'] == 0:
                self._save_checkpoint(generation + 1)

        # Phase 2: Final Validation & Reporting
        print("\n" + "="*70)
        print("SEARCH COMPLETE - GENERATING REPORT")
        print("="*70)

        self._generate_report()

    def _evaluate_strategy(self, strategy_config: Dict) -> tuple:
        """
        Evaluate a single strategy configuration.

        Args:
            strategy_config: Strategy configuration

        Returns:
            (fitness, metrics) tuple
        """
        try:
            # Train model for each validation week and backtest
            week_metrics = []

            for week_cfg in self.config['validation']['weeks_config']:
                # Get training data (before validation week)
                train_data = get_training_data_for_week(
                    self.data,
                    week_cfg,
                    lookback_days=365
                )

                if len(train_data) < 100:
                    continue

                # Normalize
                normalized_train = normalize_data(train_data.values)

                # Train model
                trainer = StrategyTrainer(strategy_config)
                model = trainer.train_model(
                    normalized_train,
                    epochs=strategy_config.get('epochs', 20)
                )

                # Get validation week data
                week_start = pd.to_datetime(week_cfg['start'])
                week_end = week_start + pd.Timedelta(days=week_cfg['days'])

                # Make timezone-aware if data index is timezone-aware
                if self.data.index.tz is not None:
                    week_start = week_start.tz_localize(self.data.index.tz)
                    week_end = week_end.tz_localize(self.data.index.tz)

                week_data = self.data[(self.data.index >= week_start) & (self.data.index < week_end)]

                if len(week_data) < 10:
                    continue

                # Generate signals
                signals = self.backtester.generate_signals(
                    model,
                    week_data,
                    confidence_threshold=strategy_config['confidence_threshold']
                )

                # Backtest
                metrics = self.backtester.backtest(
                    week_data,
                    signals,
                    stop_loss_pips=strategy_config['stop_loss_pips'],
                    take_profit_pips=strategy_config['take_profit_pips'],
                    risk_per_trade=strategy_config['risk_per_trade'],
                    trailing_stop=strategy_config.get('trailing_stop', False)
                )

                week_metrics.append(metrics)

                # Clean up
                del model
                torch.cuda.empty_cache()

            # Aggregate across weeks
            if len(week_metrics) == 0:
                return 0.0, {}

            aggregated = {
                'sharpe_ratio': np.mean([m['sharpe_ratio'] for m in week_metrics]),
                'win_rate': np.mean([m['win_rate'] for m in week_metrics]),
                'max_drawdown': np.mean([m['max_drawdown'] for m in week_metrics]),
                'total_trades': np.sum([m['total_trades'] for m in week_metrics]),
                'profit_factor': np.mean([m['profit_factor'] for m in week_metrics]),
            }

            # Calculate fitness
            fitness = calculate_composite_fitness(aggregated, self.config['fitness'])

            print(f"  Sharpe: {aggregated['sharpe_ratio']:.3f} | "
                  f"WinRate: {aggregated['win_rate']:.1%} | "
                  f"MaxDD: {aggregated['max_drawdown']:.1%} | "
                  f"Fitness: {fitness:.4f}")

            return fitness, aggregated

        except Exception as e:
            print(f"  [ERROR] Error evaluating strategy: {e}")
            return 0.0, {}

    def _save_checkpoint(self, generation: int):
        """Save checkpoint of current progress."""
        checkpoint = {
            'generation': generation,
            'best_strategy': self.best_strategy,
            'best_fitness': self.best_fitness,
            'all_results': self.all_results,
            'timestamp': datetime.now().isoformat()
        }

        checkpoint_path = f'outputs/checkpoints/checkpoint_gen_{generation}.json'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"\n[OK] Checkpoint saved: {checkpoint_path}")

    def _generate_report(self):
        """Generate final report of search results."""
        # Sort all results by fitness
        sorted_results = sorted(self.all_results, key=lambda x: x['fitness'], reverse=True)

        # Save top 50
        top_50 = sorted_results[:50]

        report_path = 'outputs/reports/search_results.json'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump({
                'search_summary': {
                    'total_strategies_tested': len(self.all_results),
                    'best_fitness': self.best_fitness,
                    'best_strategy': self.best_strategy
                },
                'top_50_strategies': top_50
            }, f, indent=2)

        print(f"\n{'='*70}")
        print("FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Total Strategies Tested: {len(self.all_results)}")
        print(f"Best Fitness: {self.best_fitness:.4f}")
        print(f"\nBest Strategy Configuration:")
        for key, value in self.best_strategy.items():
            print(f"  {key}: {value}")
        print(f"\nFull report saved to: {report_path}")
        print(f"{'='*70}\n")


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='ML Strategy Search System')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()

    print("Initializing Strategy Search System...")
    print(f"Using config: {args.config}\n")

    orchestrator = StrategySearchOrchestrator(config_path=args.config)
    orchestrator.run_search()

    print("\n*** Search complete! ***")


if __name__ == "__main__":
    main()
