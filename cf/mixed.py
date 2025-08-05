import math
from functools import lru_cache

import numpy as np
import pandas as pd

from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice, Real, Integer
from pymoo.core.mixed import MixedVariableMating
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.algorithms.moo.nsga2 import binary_tournament, RankAndCrowdingSurvival, calc_crowding_distance
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.util.randomized_argsort import randomized_argsort
from sklearn.preprocessing import StandardScaler


class CustomRankAndCrowdingSurvival(RankAndCrowdingSurvival):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scaler = StandardScaler()

    def _do(self, problem, pop, *args, n_survive=None, **kwargs):

        # get the objective space values and objects
        F = pop.get("F").astype(float, copy=False)
        F = self.scaler.fit_transform(F)

        # the final indices of surviving individuals
        survivors = []

        # do the non-dominated sorting until splitting front
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)

        for k, front in enumerate(fronts):

            # calculate the crowding distance of the front
            crowding_of_front = calc_crowding_distance(F[front, :])

            # save rank and crowding in the individual class
            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])

            # current front sorted by crowding distance if splitting
            if len(survivors) + len(front) > n_survive:
                I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
                I = I[:(n_survive - len(survivors))]

            # otherwise take the whole front unsorted
            else:
                I = np.arange(len(front))

            # extend the survivors by all or selected individuals
            survivors.extend(front[I])

        return pop[survivors]


class CustomMixedVariableMating(MixedVariableMating):

    def __init__(self,
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=None,
                 mutation=None,
                 survival=CustomRankAndCrowdingSurvival(),
                 output=MultiObjectiveOutput(),
                 repair=None,
                 eliminate_duplicates=True,
                 n_max_iterations=100,
                 **kwargs):

        super().__init__(selection=selection, crossover=crossover, mutation=mutation, survival=survival, output=output,
                         repair=repair, eliminate_duplicates=eliminate_duplicates, n_max_iterations=n_max_iterations,
                         **kwargs)

    @staticmethod
    def _get_value(problem, parent, var):
        if isinstance(parent.X, dict):
            return parent.X.get(var)
        else:
            return parent.X[problem.feature_index[var]]

    def _do(self, problem, pop, n_offsprings, parents=False, **kwargs):

        # So far we assume all crossover need the same amount of parents and create the same number of offsprings
        XOVER_N_PARENTS = 2
        XOVER_N_OFFSPRINGS = 2

        # the variables with the concrete information
        vars = problem.vars

        # group all the variables by their types
        vars_by_type = {}
        for k, v in vars.items():
            clazz = type(v)

            if clazz not in vars_by_type:
                vars_by_type[clazz] = []
            vars_by_type[clazz].append(k)

        # # all different recombinations (the choices need to be split because of data types)
        recomb = []
        for clazz, list_of_vars in vars_by_type.items():
            if clazz == Choice:
                for e in list_of_vars:
                    recomb.append((clazz, [e]))
            else:
                recomb.append((clazz, list_of_vars))

        # create an empty population that will be set in each iteration
        off = Population.new(X=[{} for _ in range(n_offsprings)])

        if not parents:
            n_select = math.ceil(n_offsprings / XOVER_N_OFFSPRINGS)
            pop = self.selection(problem, pop, n_select, XOVER_N_PARENTS, **kwargs)

        for clazz, list_of_vars in recomb:

            crossover = self.crossover[clazz]
            assert crossover.n_parents == XOVER_N_PARENTS and crossover.n_offsprings == XOVER_N_OFFSPRINGS

            try:
                _parents = [[Individual(X=np.array([parent.X[var] for var in list_of_vars])) for parent in parents] for
                            parents in pop]
            except Exception:
                _parents = [
                    [Individual(X=np.array([self._get_value(problem, parent, var) for var in list_of_vars])) for parent
                     in parents] for parents in pop]

            _vars = [vars[e] for e in list_of_vars]
            _xl, _xu = None, None

            if clazz in [Real, Integer]:
                _xl, _xu = np.array([v.bounds for v in _vars]).T

            _problem = Problem(vars=_vars, xl=_xl, xu=_xu)

            _off = crossover(_problem, _parents, **kwargs)

            mutation = self.mutation[clazz]
            _off = mutation(_problem, _off, **kwargs)

            for k in range(n_offsprings):
                for i, name in enumerate(list_of_vars):
                    off[k].X[name] = _off[k].X[i]

        for item in off:
            item.X = pd.DataFrame(item.X, index=[0]).loc[:, problem.kd_tree.one_hot_train_sequence].values[0]
        return off

    @lru_cache()
    def get_one_hot_train_sequence(self, problem):
        return problem.kd_tree.one_hot_train_sequence


class CustomMixedVariableDuplicateElimination(ElementwiseDuplicateElimination):
    def __init__(self, one_hot_train_sequence=None):
        self.one_hot_train_sequence = one_hot_train_sequence
        super().__init__()

    def is_equal(self, a, b):
        a, b = a.X, b.X
        if isinstance(a, dict) and isinstance(b, dict):
            for k, v in a.items():
                if k not in b or b.get(k) != v:
                    return False
            return True
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return True if np.array_equal(a, b) else False
        elif isinstance(a, dict) and isinstance(b, np.ndarray):
            for index, name in enumerate(self.one_hot_train_sequence):
                if a.get(name) != b[index]:
                    return False
            return True
        else:
            print("出现新的判断类型尚未处理！")
            return False
