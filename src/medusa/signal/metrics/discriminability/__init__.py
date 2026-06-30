"""Discriminability signal metrics — *how well does this feature separate classes?*
Supervised, label-relative feature scores (signal + labels -> score). Future
natural members: Fisher score, AUC, t-stat, mutual information.
"""
from medusa.signal.metrics.discriminability.signed_r2 import signed_r2

__all__ = ["signed_r2"]
