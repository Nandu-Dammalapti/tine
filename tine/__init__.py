"""Tine: Gacrux router v1 (CPU evaluation path).

Pick the cheapest model whose calibrated success probability clears a threshold
tau, where tau is certified on held-out data so quality loss versus always using
the frontier model stays at most epsilon with confidence 1 - delta.
"""

__version__ = "0.1.0"
