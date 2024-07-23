import pytest
from functools import partial
from ortools.constraint_solver import pywrapcp
from load_data.instance_type import InstanceType, process_files
from problems.vrptw import (
    create_distance_evaluator,
    add_distance_dimension,
    create_demand_evaluator,
    add_capacity_constraints,
    create_time_evaluator,
    add_time_window_constraints,
)


@pytest.fixture
def data():
    instances_data = process_files(InstanceType.VRPTW)
    data = instances_data["C1_2_1.txt"]
    return data


def test_process_files(data):
    assert len(data) > 0


def test_create_distance_evaluator(data):
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )
    distance_evaluator = create_distance_evaluator(data)
    assert distance_evaluator(manager, 0, 0) == 0
    assert distance_evaluator(manager, 0, 5) == data['vehicle_max_distance']


def test_add_distance_dimension(data):
    manager = pywrapcp.RoutingIndexManager(data['num_locations'], data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    distance_evaluator_index = routing.RegisterTransitCallback(partial(create_distance_evaluator(data)))
    add_distance_dimension(routing, manager, data, distance_evaluator_index)
    distance_dimension = routing.GetDimensionOrDie('Distance')
    assert distance_dimension.GetCumulVarSoftLowerBound(1) == 0


def test_create_demand_evaluator(data):
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )
    demand_evaluator = create_demand_evaluator(data)
    assert demand_evaluator(manager, 1) == 20


def test_add_capacity_constraints(data):
    manager = pywrapcp.RoutingIndexManager(data['num_locations'], data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    demand_evaluator_index = routing.RegisterUnaryTransitCallback(partial(create_demand_evaluator(data), manager))
    add_capacity_constraints(routing, manager, data, demand_evaluator_index)
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    assert capacity_dimension.GetCumulVarSoftLowerBound(manager.NodeToIndex(4)) == 0


def test_create_time_evaluator(data):
    time_evaluator = create_time_evaluator(data)
    manager = pywrapcp.RoutingIndexManager(data['num_locations'], data['num_vehicles'], data['depot'])
    assert round(time_evaluator(manager, 4, 8)) == round(data['service_time'][4] + data['distance_matrix'][4][8] / data['vehicle_speed'])


def test_add_time_window_constraints(data):
    manager = pywrapcp.RoutingIndexManager(data['num_locations'], data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)
    time_evaluator_index = routing.RegisterTransitCallback(partial(create_time_evaluator(data), manager))
    add_time_window_constraints(routing, manager, data, time_evaluator_index)
    time_dimension = routing.GetDimensionOrDie('Time')
    assert time_dimension.CumulVar(manager.NodeToIndex(1)).Min() == 750
    assert time_dimension.CumulVar(manager.NodeToIndex(1)).Max() == 809
