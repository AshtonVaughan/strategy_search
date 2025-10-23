"""
Main orchestrator for ML strategy search system.
Searches 1M+ trading strategies over 5-7 days using evolutionary algorithm.
"""

import yaml
import pandas as pd
import numpy as np
import torch
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List
from tqdm import tqdm

# Import modules
from search.evolution import EvolutionarySearcher
from training.trainer import StrategyTrainer
from training.model import create_model
from training.labeling import SmartLabeler
from backtesting.engine import MLBacktester, calculate_composite_fitness
from validation.cross_validator import CrossValidator
from data.collector import MultiTimeframeCollector


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
        self.data_collector = MultiTimeframeCollector(symbol=self.config['data']['symbol'])
        self.smart_labeler = SmartLabeler(
            threshold_pips=self.config.get('labeling', {}).get('threshold_pips', 20),
            lookforward_hours=self.config.get('labeling', {}).get('lookforward_hours', 12)
        )

        # Load full dataset with features
        print("Loading multi-timeframe market data with features...")
        self.data, self.feature_cols = self._load_data()

        # Results tracking
        self.all_results = []
        self.best_strategy = None
        self.best_fitness = 0

    def _load_data(self) -> tuple:
        """Load multi-timeframe data with features."""
        start = self.config['data']['start_date']
        end = self.config['data']['end_date']

        # Fetch multi-timeframe data with features
        data, feature_cols = self.data_collector.prepare_training_data(
            start_date=start,
            end_date=end,
            normalize=True
        )

        print(f"[OK] Loaded {len(data)} rows with {len(feature_cols)} features")

        return data, feature_cols

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
        Evaluate a single strategy configuration with new feature pipeline.

        Args:
            strategy_config: Strategy configuration

        Returns:
            (fitness, metrics) tuple
        """
        try:
            # Train model for each validation week and backtest
            week_metrics = []

            for week_cfg in self.config['validation']['weeks_config']:
                # Get training data with features (before validation week)
                train_data = self.data_collector.get_training_data_for_week(
                    week_start=week_cfg['start'],
                    lookback_days=180  # 6 months of training data (within Yahoo limit)
                )

                if len(train_data) < 500:  # Need more data for 80+ features
                    print(f"  [WARN] Not enough training data ({len(train_data)} rows), skipping week")
                    continue

                # Generate smart labels
                labeling_method = strategy_config.get('labeling_method', 'forward_return')
                train_data_labeled, labels = self.smart_labeler.generate_labels(
                    train_data,
                    labeling_method=labeling_method
                )

                # Check if we have enough labeled samples
                n_labeled = np.sum(labels != 0)
                if n_labeled < 50:
                    print(f"  [WARN] Not enough labeled samples ({n_labeled}), skipping week")
                    continue

                # Prepare feature matrix
                feature_matrix = train_data_labeled[self.feature_cols].values

                # Check for NaN/Inf in features BEFORE training
                if np.isnan(feature_matrix).any() or np.isinf(feature_matrix).any():
                    n_nan = np.isnan(feature_matrix).sum()
                    n_inf = np.isinf(feature_matrix).sum()
                    print(f"  [WARN] Found {n_nan} NaN and {n_inf} Inf in features, cleaning...")
                    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

                # Train model with features
                trainer = StrategyTrainer(strategy_config, input_size=len(self.feature_cols))
                model = trainer.train_model_with_labels(
                    features=feature_matrix,
                    labels=labels,
                    epochs=strategy_config.get('epochs', 20)
                )

                # Get validation week data with features
                week_start = pd.to_datetime(week_cfg['start'])
                week_end = week_start + timedelta(days=week_cfg['days'])

                # Data is timezone-naive now
                week_data = self.data[(self.data.index >= week_start) & (self.data.index < week_end)]

                if len(week_data) < 10:
                    continue

                # Generate signals using features
                signals = self.backtester.generate_signals_with_features(
                    model,
                    week_data,
                    self.feature_cols,
                    confidence_threshold=strategy_config['confidence_threshold'],
                    strategy_logic=strategy_config.get('strategy_logic', 'simple')
                )

                # Debug: Check signals
                n_buy = (signals['signal'] == 1).sum()
                n_sell = (signals['signal'] == -1).sum()
                print(f"  Signals: {n_buy} buys, {n_sell} sells")

                # Backtest
                metrics = self.backtester.backtest(
                    week_data,
                    signals,
                    stop_loss_pips=strategy_config['stop_loss_pips'],
                    take_profit_pips=strategy_config['take_profit_pips'],
                    risk_per_trade=strategy_config['risk_per_trade'],
                    trailing_stop=strategy_config.get('trailing_stop', False)
                )

                # Debug: Check metrics
                print(f"  Trades: {metrics.get('total_trades', 0)}, WinRate: {metrics.get('win_rate', 0):.1%}")

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
                  f"Trades: {aggregated['total_trades']} | "
                  f"Fitness: {fitness:.4f}")

            return fitness, aggregated

        except Exception as e:
            print(f"  [ERROR] Error evaluating strategy: {e}")
            import traceback
            traceback.print_exc()
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

        if self.best_strategy is not None:
            print(f"\nBest Strategy Configuration:")
            for key, value in self.best_strategy.items():
                print(f"  {key}: {value}")
        else:
            print(f"\n[WARN] No successful strategies found!")
            print(f"  All strategies had fitness 0.0 (likely due to errors)")

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
