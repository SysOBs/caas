import numpy as np
from pymoo.optimize import minimize

from caas.definitions import (Workflow, build_service, max_system_cost,
                              system_cost, workflow_availabilities)
from caas.optimizer import Optimizer
from json_utils import load_gglf_configs
from pymoo_utils import ga_algorithm


def run():
    """
    This method contains a simple example of how to use the CAAS to optimize 
    service replicas.

    The example is based on the Robot Shop application.
    First we load the GLF configurations.
    Then we build the services with the GLF configurations.
    Then we build the workflows with the services.
    Then we define the load factors matrix.
    And finally we run the optimizer.

    The optimizer will try to find the optimal number of replicas for each
    service in order to maximize the availability of the workflows while
    minimizing the cost of the system.
    """

    gglf_configs_robot_shop = load_gglf_configs(
        "./gglf_configs_robot_shop.json")

    web_svc = build_service(
        "web", gglf_configs_robot_shop, cpu=1, mem=2, replicas=1)
    catalogue_svc = build_service(
        "catalogue", gglf_configs_robot_shop, cpu=1, mem=2, replicas=1)
    user_svc = build_service(
        "user", gglf_configs_robot_shop, cpu=1, mem=2, replicas=1)
    cart_svc = build_service(
        "cart", gglf_configs_robot_shop, cpu=1, mem=2, replicas=1)
    ratings_svc = build_service(
        "ratings", gglf_configs_robot_shop, cpu=1, mem=2, replicas=1)

    LOADS = [1, 100, 300, 500]

    MAX_CPU = 32  # vCPUs
    MAX_MEM = 128  # GiB memory

    SLO = 0.99

    wf1 = Workflow([web_svc, catalogue_svc])
    wf2 = Workflow([web_svc, user_svc, catalogue_svc, cart_svc])
    wf3 = Workflow([web_svc, catalogue_svc, ratings_svc])

    m = np.matrix([[1, 1, 0, 0], [0.2, 0.2, 0.2, 0.2], [0, 0, 0, 0]])

    system_services = [web_svc, user_svc, catalogue_svc, cart_svc, ratings_svc]
    system_workflows = [wf1, wf2, wf3]
    for load in LOADS:
        problem = Optimizer(system_services, system_workflows, load,
                            m, MAX_CPU, MAX_MEM, SLO)
        res = minimize(problem, ga_algorithm, ("n_gen", 100), seed=1,
                       save_history=True, verbose=False)
        optimal_config = res.X
        F = res.F
        CV = res.CV

        if optimal_config is None:
            print(f"Load {load} -> config: None -> u=None, c=None -> f:None")
            continue

        for i, service in enumerate(system_services):
            service.replicas = optimal_config[i]

        wf1 = Workflow([web_svc, catalogue_svc])
        wf2 = Workflow([web_svc, user_svc, catalogue_svc, cart_svc])
        wf3 = Workflow([web_svc, catalogue_svc, ratings_svc])
        system_workflows = [wf1, wf2, wf3]

        wf_unavailabilities = workflow_availabilities(
            system_workflows, load, m)
        cost = system_cost(system_services) / max_system_cost(MAX_CPU, MAX_MEM)
        print(
            f"Load {load} -> config: {optimal_config} -> wf_u(s)={1-wf_unavailabilities}, c={cost} -> f:{F}")


if __name__ == "__main__":
    run()
