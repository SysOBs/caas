import numpy
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

ga_algorithm = GA(
    pop_size=100,
    n_offsprings=10,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=1.0, eta=15, vtype=int, repair=RoundingRepair()),
    mutation=PM(prob=1.0, eta=3, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True,
)

ps_algorithm = PatternSearch(x0=numpy.array([1, 1]))

de_algorithm = DE(
    pop_size=100,
    n_offsprings=10,
    sampling=IntegerRandomSampling(),
    # variant="DE/best/1/bin",
    # CR=0.3,
    # dither="vector",
    # jitter=False
)

isres_algorithm = ISRES(n_offsprings=200, rule=1.0 /
                        7.0, gamma=0.85, alpha=0.2)
