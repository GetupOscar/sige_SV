import torch
import numpy as np
from itertools import combinations

class ShapleyExplainer:
    def __init__(self, model):
        self.model = model
        self.feature_names = [
            'user_mec_distance',
            'mec_compute_capacity',
            'user_position',
            'bandwidth',
            'qoe'
        ]
        
    def perturb_features(self, state, active_features):
        """
        Perturb features by masking inactive ones with random noise
        """
        perturbed = state.clone()
        mask = torch.ones_like(perturbed)
        mask[:, active_features] = 0
        noise = torch.randn_like(perturbed) * 0.1 * perturbed.std()
        return perturbed * (1-mask) + noise * mask

    def calculate_marginal(self, state, subset, feature):
        """
        Calculate marginal contribution of a feature given a subset
        """
        # Get Q-values without the feature
        subset_state = self.perturb_features(state, subset)
        q_without = self.model(subset_state).max(1)[0]
        
        # Get Q-values with the feature
        with_feature = subset + [feature]
        full_state = self.perturb_features(state, with_feature)
        q_with = self.model(full_state).max(1)[0]
        
        return (q_with - q_without).item()

    def explain(self, state):
        """
        Calculate Shapley values for each feature
        """
        n_features = state.size(1)
        shapley_values = {}
        
        # Calculate Shapley value for each feature
        for i in range(n_features):
            marginal_contributions = []
            
            # Consider all possible feature subsets
            for r in range(n_features):
                for subset in combinations([j for j in range(n_features) if j != i], r):
                    subset = list(subset)
                    # Calculate marginal contribution
                    marginal = self.calculate_marginal(state, subset, i)
                    # Weight by number of possible permutations
                    weight = 1.0 / (n_features * np.math.comb(n_features-1, len(subset)))
                    marginal_contributions.append(marginal * weight)
            
            # Store Shapley value for this feature
            shapley_values[self.feature_names[i]] = sum(marginal_contributions)
            
        return shapley_values

    def plot_feature_importance(self, values, ax=None):
        """
        Plot feature importance as a bar chart
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        features = list(values.keys())
        importances = list(values.values())
        
        # Sort by absolute importance
        sorted_idx = np.argsort(np.abs(importances))
        pos = np.arange(len(features)) + .5
        
        ax.barh(pos, [importances[i] for i in sorted_idx])
        ax.set_yticks(pos)
        ax.set_yticklabels([features[i] for i in sorted_idx])
        ax.set_xlabel('Shapley Value (Feature Importance)')
        ax.set_title('Feature Importance Analysis')
        
        plt.tight_layout()
        return ax
