"""
edcsl_loss.py
=============
Example-Dependent Cost-Sensitive Loss function for XGBoost.
Based on: Bahnsen et al. (2014)

The script defines the mathematical objective function that replaces 
the standard cost-blind log-loss in XGBoost.
"""

import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    The function calculates the numerically stable sigmoid.
    It clips the input to prevent the exponential calculation from overflowing.
    """
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def build_edcsl_objective(amounts: np.ndarray, y_true: np.ndarray, Ca: float = 50.0):
    """
    The function builds and returns the custom objective function.
    XGBoost uses this objective to apply cost-sensitive learning at every training step.
    """
    amounts = np.array(amounts, dtype=np.float64)
    y_true  = np.array(y_true,  dtype=np.float64)

    # ── THE LOGARITHMIC FIX ──
    # We apply log1p (natural log of 1 + x) to compress extreme financial outliers.
    # This prevents 'Gradient Explosion' where a R10,000,000 transaction creates 
    # a gradient so large it destroys the model's decision boundary.
    scaled_amounts = np.log1p(amounts)
    scaled_Ca = np.log1p(Ca)

    # Assign the logarithmically scaled financial weight to each transaction.
    cost_weights = np.where(y_true == 1, scaled_amounts, scaled_Ca)

    # Ensure no weight falls below the scaled administrative cost threshold.
    cost_weights = np.clip(cost_weights, scaled_Ca, None)

    # Normalize the weights by dividing by the mean.
    # This keeps the overall magnitude of the gradients aligned with XGBoost's default learning rate.
    mean_weight = np.mean(cost_weights)
    normalized_weights = cost_weights / mean_weight

    def edcsl_objective(y_pred: np.ndarray, dtrain) -> tuple:
        """
        The inner function calculates the specific gradients and hessians for XGBoost.
        """
        # Convert raw predictions into probabilities (0 to 1).
        p = sigmoid(y_pred)
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # GRADIENT: The direction to adjust the rules, multiplied by the financial weight.
        grad = (p - y_true) * normalized_weights

        # HESSIAN: The step size for the adjustment, multiplied by the financial weight.
        hess = p * (1 - p) * normalized_weights

        # XGBoost requires strictly positive hessians to remain mathematically stable.
        hess = np.clip(hess, 1e-7, None)

        return grad, hess

    return edcsl_objective