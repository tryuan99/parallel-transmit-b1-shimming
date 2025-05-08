"""The B1 shimming optimization problem finds the optimal relative magnitudes
and phases for the TX coils to create a uniform B1 field within the region of
interest.
"""

import numpy as np

from optimization.problem import Problem
from simulation.b_field import BField


class B1ShimmingMagnitudeOptimizationProblem(Problem):
    """B1 field shimming magnitude optimization problem.

    Attributes:
        fields: B field maps of each TX coil.
    """

    def __init__(self, fields: list[BField]) -> None:
        self.fields = fields

    def num_coils(self) -> int:
        """Returns the number of TX coils."""
        return len(self.fields)

    def num_variables_per_coil(self) -> int:
        """Returns the number of design variables per TX coil."""
        # We can set the relative magnitude and phase of each coil.
        return 2

    def num_variables(self) -> int:
        """Returns the number of design variables."""
        return self.num_coils() * self.num_variables_per_coil()

    def lower_bound(self) -> np.ndarray:
        """Returns the lower bound on the design variables."""
        return np.tile(self.lower_bound_per_coil(), self.num_coils())

    def lower_bound_per_coil(self) -> np.ndarray:
        """Returns the lower bound on the design variables for a single TX
        coil.
        """
        relative_magnitude_lower_bound = 0.1
        phase_lower_bound = 0
        return np.array([relative_magnitude_lower_bound, phase_lower_bound])

    def upper_bound(self) -> np.ndarray:
        """Returns the upper bound on the design variables."""
        return np.tile(self.upper_bound_per_coil(), self.num_coils())

    def upper_bound_per_coil(self) -> np.ndarray:
        """Returns the upper bound on the design variables for a single TX
        coil.
        """
        relative_magnitude_upper_bound = 1
        phase_upper_bound = 2 * np.pi
        return np.array([relative_magnitude_upper_bound, phase_upper_bound])

    def num_objectives(self) -> int:
        """Returns the number of objectives."""
        return 1

    def evaluate_objectives(self, x: np.ndarray) -> list[float]:
        """Evaluates the objective(s) on the given design variable values.

        Args:
            x: Design variable values.

        Returns:
            The objective(s) evaluated on the given design variable values.
        """
        relative_magnitudes = x[::self.num_variables_per_coil()]
        phases = x[1::self.num_variables_per_coil()]

        b_field_sum = BField(
            coordinates=self.fields[0].coordinates,
            data=np.sum(
                [
                    relative_magnitudes[coil_index] *
                    np.exp(1j * phases[coil_index]) *
                    self.fields[coil_index].data
                    for coil_index in range(self.num_coils())
                ],
                axis=0,
            ),
        )
        b1_inhomogeneity = b_field_sum.calculate_b1_inhomogeneity()
        return b1_inhomogeneity
