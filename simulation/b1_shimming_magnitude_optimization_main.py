"""Optimizes the antenna array, such that the antenna array elements lie on the
given surface.
"""

import numpy as np
import pymoo.operators.sampling.lhs
from absl import app, flags, logging

from optimization.pso_single_objective_optimizer import \
    PsoSingleObjectiveOptimizer
from simulation.b1_shimming_optimization import \
    B1ShimmingMagnitudeOptimizationProblem
from simulation.b_field import BField
from simulation.field_file import HFieldFile, SarFieldFile

FLAGS = flags.FLAGS


def optimize_b1_shims(
    fields: list[BField],
    population_size: int,
    num_generations: int,
    seed: int = None,
) -> None:
    """Optimizes the B1 shims over the region of interest.

    Args:
        fields: B field maps of each TX coil.
        population_size: Population size.
        num_generations: Number of generations.
        seed: Random seed.
    """
    problem = B1ShimmingMagnitudeOptimizationProblem(fields)
    optimizer = PsoSingleObjectiveOptimizer(
        problem,
        population_size,
        num_generations,
        seed,
    )
    optimizer.run(sampling=pymoo.operators.sampling.lhs.LHS())

    # Log the B1 inhomogeneity and the B1 shim values.
    logging.info("B1 inhomogeneity: %f", optimizer.objective_value)
    logging.info("B1 shims: %s", optimizer.optimal_value)

    # Output the B1 shim values in a CSV format.
    logging.info("Coil index,Relative magnitude,Phase")
    for coil_index in range(problem.num_coils()):
        logging.info(
            "%d,%f,%f",
            coil_index + 1,
            optimizer.optimal_value[problem.num_variables_per_coil() *
                                    coil_index],
            optimizer.optimal_value[problem.num_variables_per_coil() *
                                    coil_index + 1],
        )

    # Plot the resulting B1 field magnitude.
    relative_magnitudes = (
        optimizer.optimal_value[::problem.num_variables_per_coil()])
    phases = optimizer.optimal_value[1::problem.num_variables_per_coil()]
    weights = relative_magnitudes * np.exp(1j * phases)
    b_field = BField.sum(fields, weights)
    b_field.plot_b1_magnitude()
    b_field.plot_normalized_b1_magnitude()


def main(argv):
    assert len(argv) == 1, argv

    mask = SarFieldFile(FLAGS.mask)
    fields = [BField(HFieldFile(field), mask) for field in FLAGS.fields]
    optimize_b1_shims(
        fields,
        FLAGS.population_size,
        FLAGS.num_generations,
        FLAGS.seed,
    )


if __name__ == "__main__":
    flags.DEFINE_multi_string(
        "fields",
        [
            "simulation/data/parallel_tx_phantom_b1_map_1.fld",
            "simulation/data/parallel_tx_phantom_b1_map_2.fld",
            "simulation/data/parallel_tx_phantom_b1_map_3.fld",
            "simulation/data/parallel_tx_phantom_b1_map_4.fld",
            "simulation/data/parallel_tx_phantom_b1_map_5.fld",
            "simulation/data/parallel_tx_phantom_b1_map_6.fld",
            "simulation/data/parallel_tx_phantom_b1_map_7.fld",
            "simulation/data/parallel_tx_phantom_b1_map_8.fld",
            "simulation/data/parallel_tx_phantom_b1_map_9.fld",
            "simulation/data/parallel_tx_phantom_b1_map_10.fld",
            "simulation/data/parallel_tx_phantom_b1_map_11.fld",
            "simulation/data/parallel_tx_phantom_b1_map_12.fld",
            "simulation/data/parallel_tx_phantom_b1_map_13.fld",
            "simulation/data/parallel_tx_phantom_b1_map_14.fld",
            "simulation/data/parallel_tx_phantom_b1_map_15.fld",
            "simulation/data/parallel_tx_phantom_b1_map_16.fld",
        ],
        "H field map files for each coil.",
    )
    flags.DEFINE_string("mask", "simulation/data/parallel_tx_phantom_sar.fld",
                        "SAR field file.")
    flags.DEFINE_integer("population_size",
                         100,
                         "Population size.",
                         lower_bound=1)
    flags.DEFINE_integer("num_generations",
                         200,
                         "Number of generations.",
                         lower_bound=1)
    flags.DEFINE_integer("seed", None, "Random seed.")

    app.run(main)
