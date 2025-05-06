"""The particle swarm optimization algorithm is a single-objective optimization
problem solver to uses a swarm of particles to guide its search.
"""

import pymoo.algorithms.soo.nonconvex.pso
import pymoo.optimize

from optimization.problem import Problem, ProblemWrapper
from optimization.single_objective_optimizer import SingleObjectiveOptimizer


class PsoSingleObjectiveOptimizer(SingleObjectiveOptimizer):
    """Particle swarm optimization algorithm.

    Attributes:
        population_size: Population size.
        num_generations: Number of generations.
        seed: Random seed.
    """

    def __init__(
        self,
        problem: Problem,
        population_size: int = 25,
        num_generations: int = 200,
        seed: int = None,
    ) -> None:
        super().__init__(problem)
        self.population_size = population_size
        self.num_generations = num_generations
        self.seed = seed

    def run(self, *args, **kwargs) -> None:
        """Solves the optimization problem.

        Args:
            args: Additional arguments.
            kwargs: Additional keyword arguments.
        """
        algorithm = pymoo.algorithms.soo.nonconvex.pso.PSO(
            pop_size=self.population_size,
            *args,
            **kwargs,
        )
        result = pymoo.optimize.minimize(
            ProblemWrapper(self.problem),
            algorithm,
            termination=("n_gen", self.num_generations),
            seed=self.seed,
            verbose=True,
        )
        self.optimal_value = result.X
        self.objective_value = result.F
