import torch
import numpy as np
from itertools import combinations
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os

class ShapleyExplainer:
    def __init__(self, model):
        self.model = model
        self.feature_names = [
            "user_mec_distance",
            "MEC_process_ability",
            "user_position_x",
            "user_position_y",
            "bandwidth"
        ]
        self.history = []
        
    def explain(self, state):
        """Calculate Shapley values for each feature"""
        n_features = len(self.feature_names)
        shapley_values = {name: 0.0 for name in self.feature_names}
        baseline = torch.zeros_like(state)
        
        # Calculate value with all features
        full_value = self.model(state).max().item()
        
        # Calculate marginal contributions
        for k in range(1, n_features + 1):
            for subset in combinations(range(n_features), k):
                # Create mask for this subset
                mask = torch.zeros_like(state)
                for idx in subset:
                    mask[0][idx] = 1
                
                # Calculate value with this subset
                subset_value = self.model(state * mask).max().item()
                
                # Calculate value without each feature in subset
                for idx in subset:
                    mask_without = mask.clone()
                    mask_without[0][idx] = 0
                    value_without = self.model(state * mask_without).max().item()
                    
                    # Update Shapley value
                    marginal = subset_value - value_without
                    weight = self._calculate_weight(k-1, n_features)
                    shapley_values[self.feature_names[idx]] += marginal * weight
        
        # Store results with timestamp and raw state data
        self._store_result(shapley_values, state)
        
        return shapley_values
    
    def _calculate_weight(self, size_subset, n):
        """Calculate the Shapley kernel weight"""
        return 1.0 / (n * np.math.comb(n-1, size_subset))
    
    def _store_result(self, shapley_values, state):
        """Store Shapley values with raw state data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        result = {
            "metadata": {
                "timestamp": timestamp,
                "sampling_interval": 5
            },
            "raw_data": {
                "user_locations": {
                    "x": state[0][2].item(),
                    "y": state[0][3].item()
                },
                "bandwidth": state[0][4].item(),
                "mec_distance": state[0][0].item(),
                "process_ability": state[0][1].item()
            },
            "shapley_values": shapley_values
        }
        
        self.history.append(result)
        
        # Keep only last 10 results
        if len(self.history) > 10:
            self.history = self.history[-10:]
    
    def plot_summary(self, with_error=False, save_path=None):
        """Plot Shapley values summary"""
        if not self.history:
            return
            
        # Prepare data
        features = self.feature_names
        values = {f: [] for f in features}
        
        for entry in self.history:
            for f in features:
                values[f].append(entry["shapley_values"][f])
        
        # Calculate means and std
        means = {f: np.mean(values[f]) for f in features}
        stds = {f: np.std(values[f]) for f in features}
        
        # Sort features by absolute mean value
        sorted_features = sorted(features, key=lambda x: abs(means[x]), reverse=True)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Bar positions
        x = np.arange(len(features))
        
        # Plot bars
        bars = plt.bar(x, [means[f] for f in sorted_features])
        
        # Color coding
        for i, bar in enumerate(bars):
            if means[sorted_features[i]] > 0:
                bar.set_color('#4C72B0')  # Blue for positive
            else:
                bar.set_color('#DD8452')  # Orange for negative
        
        # Add error bars if requested
        if with_error:
            plt.errorbar(x, [means[f] for f in sorted_features],
                        yerr=[stds[f] for f in sorted_features],
                        fmt='none', color='#404040', capsize=4)
        
        plt.xticks(x, sorted_features, rotation=45, ha='right')
        plt.title('Feature Importance Analysis' + 
                 (' (with std dev)' if with_error else ''))
        plt.xlabel('Features')
        plt.ylabel('Shapley Value')
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            save_path = os.path.join(os.getcwd(), f'shapley_summary{"_with_error" if with_error else ""}_{timestamp}.png')
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        plt.savefig(save_path)
    
    def save_history(self, filepath=None):
        """Save analysis history to JSON file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filepath = os.path.join(os.getcwd(), f'shapley_analysis_{timestamp}.json')
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory:  # Only create directory if path has a directory component
            os.makedirs(directory, exist_ok=True)
        
        # Convert numpy values to Python native types
        history_copy = []
        for entry in self.history:
                    },
                    "bandwidth": float(entry["raw_data"]["bandwidth"]),
                    "mec_distance": float(entry["raw_data"]["mec_distance"]),
                    "process_ability": float(entry["raw_data"]["process_ability"])
                },
                "shapley_values": {k: float(v) for k, v in entry["shapley_values"].items()}
            }
            history_copy.append(entry_copy)
        
        with open(filepath, 'w') as f:
            json.dump(history_copy, f, indent=2)
