from abc import ABC, abstractmethod

import numpy as np

from caas import glf


class AvailabilityFunction(ABC):

    @abstractmethod
    def calc(self):
        pass


class GeneralisedLogisticFunction(AvailabilityFunction):
    def __init__(self, b: float, v: float, q: float, a: float = 1, k: float = 0, c: float = 1):
        self.a = a
        self.k = k
        self.c = c
        self.b = b
        self.v = v
        self.q = q

    def calc(self, t: np.ndarray) -> np.ndarray:
        return glf.generalised_logistic_function(t=t, a=self.a, k=self.k, c=self.c, b=self.b, v=self.v, q=self.q)


class GGLFunction(AvailabilityFunction):
    def __init__(self, a: float, k: float, c: float, b: float, v: float, q: float):
        self.a = a
        self.k = k
        self.c = c
        self.b = b
        self.v = v
        self.q = q

    def calc(self, t: np.ndarray, n: np.ndarray, s: np.ndarray) -> np.ndarray:
        X = t, n, s
        return glf.g_generalised_logistic_function(X, a=self.a, k=self.k, c=self.c, b=self.b, v=self.v, q=self.q)


class PerfectAvailabilityFunction(AvailabilityFunction):

    def calc(self, param1, param2, param3) -> float:
        return 1.0


class Service:
    def __init__(self, name: str, cpu: float, mem: float,
                 replicas: int, availability_func: AvailabilityFunction,
                 machine_availability: float, p: float):
        self.name = name

        self.cpu = cpu
        self.memory = mem

        self.replicas = replicas

        self.availability_func = availability_func
        self.machine_availability = machine_availability

        self.p = p

    def __repr__(self) -> str:
        return f"name={self.name}, cpu={self.cpu}, mem={self.memory}, replicas={self.replicas}"

    def __str__(self) -> str:
        return f"name={self.name}, cpu={self.cpu}, mem={self.memory}, replicas={self.replicas}"

    def calc_availability(self, load: float) -> float:
        return self.availability_func.calc(load)


def build_service(service_name: str, gglf_configs: dict, cpu: float,
                  mem: float, replicas: int = 1,
                  machine_availability: float = 1) -> Service:
    # order: a: float, k: float, c: float, b: float, v: float, q:
    p = gglf_configs[service_name]["amdahl_p"]
    a = gglf_configs[service_name]["popt"][0]
    k = gglf_configs[service_name]["popt"][1]
    c = gglf_configs[service_name]["popt"][2]
    b = gglf_configs[service_name]["popt"][3]
    v = gglf_configs[service_name]["popt"][4]
    q = gglf_configs[service_name]["popt"][5]
    return Service(name=service_name, cpu=cpu, mem=mem, replicas=replicas,
                   availability_func=GGLFunction(a=a, k=k, c=c, b=b, v=v, q=q),
                   machine_availability=machine_availability, p=p)


def round_robin_availability(service: Service, rps: np.float64) -> np.float64:
    """
    Args:
        service (Service): Service instance
        replicas (int): Number of replicas
        rps (np.float64): Requests per second
    Returns:
        np.float64: Availability of the service
    """
    if rps == 0:
        return np.float64(1)
    if service.replicas == 0:
        return np.float64(0)
    return service.availability_func.calc(rps, service.replicas, service.p)


def serial_availability(availabilities: np.ndarray) -> np.float64:
    if len(availabilities) == 0:
        return np.float64(0)
    return np.prod(availabilities)  # type: ignore


def system_availability(services: list[Service], replicas: np.ndarray, load: float, load_factors: np.ndarray):
    availabilities = []

    for i, service in enumerate(services):
        availabilities.append(service.machine_availability)
        availabilities.append(round_robin_availability(
            service, replicas[i], load * load_factors[i]))

    return serial_availability(np.array(availabilities))


def system_cost(services: list[Service],
                cpu_coefficient: float = 0.0427,
                mem_coefficient: float = 0.0039) -> float:
    replicas = np.array([])
    cpu = np.array([])
    mem = np.array([])
    for service in services:
        replicas = np.append(replicas, service.replicas)
        cpu = np.append(cpu, service.cpu)
        mem = np.append(mem, service.memory)
    return np.sum(replicas * (cpu * cpu_coefficient + mem * mem_coefficient))


def max_system_cost(cpus: float, mem: float, cpu_c: float = 0.0427, mem_c: float = 0.0039) -> float:
    return cpu_c * cpus + mem_c * mem


class Workflow:

    def __init__(self, services: list[Service]) -> None:
        """
        Args:
            services (list[Service]): List of services in the workflow
        """
        self.services = services

    def __repr__(self) -> str:
        return f"{self.services}"

    def __str__(self) -> str:
        return f"{self.services}"

    def calc_availability(self, load: float,
                          load_factors: np.ndarray) -> float:
        """
        Args:
            load (np.float64): Load at the entry point of the system
            load_factors (np.ndarray): Load factors for each service in the
            workflow
        Returns:
            np.float64: Availability of the workflow
        """
        availabilities = np.array([])

        for i, service in enumerate(self.services):
            availabilities = np.append(availabilities,
                                       service.machine_availability)
            service_availability = round_robin_availability(service,
                                                            load * load_factors[i])
            availabilities = np.append(availabilities, service_availability)

        availability = serial_availability(availabilities)

        if availability > 1:
            return 1
        elif availability < 0:
            return 0
        return availability


def workflow_availabilities(workflows: list[Workflow], load: float,
                            load_factors_matrix: np.matrix) -> np.ndarray:
    workflow_avls = np.array([])
    for i, workflow in enumerate(workflows):
        load_factors = np.squeeze(np.asarray(load_factors_matrix[i]))
        workflow_avl = workflow.calc_availability(load, load_factors)
        workflow_avls = np.append(workflow_avls, workflow_avl)
    return workflow_avls
