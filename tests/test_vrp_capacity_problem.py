import pytest
from ortools.constraint_solver import pywrapcp
from load_data.instance_type import process_files, InstanceType


@pytest.fixture
def instances_data():
    return process_files(InstanceType.VRPTW)


def test_process_files(instances_data):
    assert len(instances_data) > 0


def test_create_routing_index_manager(instances_data):
    for instance, data in instances_data.items():
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        assert manager.GetNumberOfNodes() == len(data["distance_matrix"])
        assert manager.GetNumberOfVehicles() == data["num_vehicles"]


def test_distance_callback():
    for instance, data in process_files(InstanceType.VRPTW).items():
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        for vehicle in range(data["num_vehicles"]):
            routing.SetArcCostEvaluatorOfVehicle(transit_callback_index, vehicle)
        assert distance_callback(0, 5) == data["distance_matrix"][0][5]
