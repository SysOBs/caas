from math import ceil

import numpy as np
from pymoo.core.problem import Problem

from caas.definitions import (Service, Workflow, max_system_cost,
                              system_availability, system_cost,
                              workflow_availabilities)

# from svc_availability_definitions import system_availability
# from svc_availability_definitions_v2 import (Service, Workflow,
#                                             max_system_cost, system_cost,
#                                             workflow_availabilities)


class Optimizer(Problem):

    def __init__(self, services: list[Service], workflows: list[Workflow],
                 load: np.float64, load_factors_matrix: np.matrix,
                 max_cpu: np.float64, max_mem: np.float64, slo: np.float64,
                 weight_1: np.float64 = np.float64(1.0),
                 weight_2: np.float64 = np.float64(1.0)):
        self.services = services
        self.workflows = workflows

        self.load = load
        self.load_factors_matrix = load_factors_matrix

        self.max_cpu = max_cpu
        self.max_mem = max_mem

        self.slo = slo

        self.weight_1 = weight_1
        self.weight_2 = weight_2

        self.cpu = np.array([])
        self.mem = np.array([])
        for service in self.services:
            self.cpu = np.append(self.cpu, service.cpu)
            self.mem = np.append(self.mem, service.memory)

        self.max_replicas = max(ceil(self.max_cpu / min(self.cpu)),
                                ceil(self.max_mem / min(self.mem)))
        self.max_cost = max_system_cost(self.max_cpu, self.max_mem)

        super().__init__(n_var=len(self.services), n_obj=1,
                         n_constr=3, xl=0, xu=self.max_replicas, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        availabilities = np.array([])
        costs = np.array([])
        for x_row in x:
            for i, service in enumerate(self.services):
                service.replicas = x_row[i]
            wf_avls = workflow_availabilities(
                self.workflows, self.load, self.load_factors_matrix)
            costs = np.append(costs, system_cost(self.services))
            cost = costs / self.max_cost
            availabilities = np.append(availabilities, np.prod(wf_avls))

        unavailability = 1 - availabilities

        out["F"] = self.weight_1 * unavailability + self.weight_2 * cost

        out["G"] = np.column_stack([
            - (1 - unavailability - self.slo),
            # TODO: Check if this is correct
            # numpy.sum(self.cpu_counts * x, axis=1) +
            # numpy.sum(self.mem_counts * x, axis=1) - 1,
            np.sum(self.cpu * x, axis=1) - self.max_cpu,
            np.sum(self.mem * x, axis=1) - self.max_mem,
        ])
