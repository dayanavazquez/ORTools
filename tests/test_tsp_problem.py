import pytest
from unittest.mock import patch
from ortools.constraint_solver import pywrapcp
from load_data.instance_type import process_files, InstanceType
from problems.tsp.execute.tsp_problem import compute_euclidean_distance_matrix, save_solution, execute


@pytest.fixture
def data():
    instances_data = process_files(InstanceType.TSP)
    data = instances_data["att48.txt"]
    return data


def test_compute_euclidean_distance_matrix(data):
    distance_matrix = compute_euclidean_distance_matrix(data["locations"])
    assert distance_matrix[0][0] == 0
    assert distance_matrix[0][1] == 4726
    assert distance_matrix[0][2] == 1204
    assert distance_matrix[0][3] == 6362


@patch("problems.tsp.execute.tsp_problem.os.makedirs")
@patch("problems.tsp.execute.tsp_problem.open")
def test_save_solution(mock_open, mock_makedirs, data):
    manager = pywrapcp.RoutingIndexManager(
        len(data["locations"]),
        data["num_vehicles"],
        data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)
    solution = routing.SolveWithParameters(pywrapcp.DefaultRoutingSearchParameters())
    save_solution(manager, routing, solution, "att48.txt")
    mock_makedirs.assert_called_once_with("../solutions", exist_ok=True)
    mock_open.assert_called_with('../solutions/solution_att48.txt', 'w')
