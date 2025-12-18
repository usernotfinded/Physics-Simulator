# MIT License (see LICENSE)
"""
Physical constants used throughout the simulation.

These constants use SI units and represent fundamental physical values
needed for electromagnetic force calculations.
"""
from __future__ import annotations

# Coulomb's constant (electrostatic constant), k = 1/(4πε₀)
# Value: 8.9875517923 × 10⁹ N·m²/C²
# Reference: https://physics.nist.gov/cgi-bin/cuu/Value?k
K_COULOMB: float = 8.9875517923e9

# Default softening parameter for Coulomb force to prevent singularity at r→0.
# When two charged particles get very close, the force would approach infinity;
# this epsilon regularizes the denominator: r² → r² + ε².
DEFAULT_EPS: float = 1e-3
