"""Optimizes the antenna array, such that the antenna array elements lie on the
given surface.
"""

import pymoo.operators.sampling.lhs
from absl import app, flags, logging

from optimization.pso_single_objective_optimizer import \
    PsoSingleObjectiveOptimizer
from simulation.b1_field_data import B1FieldData
from simulation.b1_shimming_optimization import \
    B1ShimmingMagnitudeOptimizationProblem
from simulation.field_file import HFieldFile, SarFieldFile

FLAGS = flags.FLAGS


def optimize_b1_shims(
    fields: list[HFieldFile],
    population_size: int,
    num_generations: int,
    seed: int = None,
) -> None:
    """Optimizes the B1 shims over the region of interest.

    Args:
        fields: B1 field maps of each TX coil.
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
            "%d,%f,%f", coil_index + 1,
            optimizer.optimal_value[problem.num_variables_per_coil() *
                                    coil_index],
            optimizer.optimal_value[problem.num_variables_per_coil() *
                                    coil_index + 1])


def main(argv):
    assert len(argv) == 1, argv

    mask = SarFieldFile(FLAGS.mask)
    fields = [B1FieldData(HFieldFile(field), mask) for field in FLAGS.fields]
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
            "simulation/data/birdcage_phantom_7T.fld",
            "simulation/data/birdcage_phantom_7T.fld",
        ],
        "H field map files for each coil.",
    )
    flags.DEFINE_string("mask", "simulation/data/birdcage_phantom_sar.fld",
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
