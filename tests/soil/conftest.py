"""Fixtures for soil API tests."""

from typing import Any, Union
from unittest import mock

import pandas as pd
import pytest

from owi.metadatabase.soil import SoilAPI


@pytest.fixture(scope="function")
def soil_init(header: dict[str, str]) -> dict[str, Any]:
    """Fixture providing expected initialization values for SoilAPI."""
    return {
        "api_root": "https://owimetadatabase.azurewebsites.net/api/v1/soildata/",
        "header": header,
        "auth": None,
        "uname": None,
        "password": None,
    }


@pytest.fixture(scope="function")
def api_soil(header: dict[str, str]) -> SoilAPI:
    """Fixture providing a SoilAPI instance for testing."""
    return SoilAPI(header=header)


@pytest.fixture
def mock_requests_get_proximity_entities_2d(mocker: mock.Mock) -> mock.Mock:
    """Mock for proximity entity requests."""
    mock_obj = mocker.patch("owi.metadatabase.soil.io.API.process_data")

    def mock_process_data(url_data_type, url_params, output_type):
        # Check for wrong parameters (latitude without decimal part)
        if url_params.get("latitude") == 50:
            raise ValueError("Invalid latitude parameter")

        df = pd.DataFrame(
            {
                "col_1": pd.array([11, 21]),
                "col_2": pd.array([12.5, 22.5]),
                "col_3": pd.array([True, False]),
                "col_4": pd.array(["test", "test2"], dtype=object),
                "col_5": [{"key": "value"}, {"key2": "value2"}],
                "col_6": pd.array([[1, 2, 3], [4, 5, 6]], dtype=object),
            }
        )
        df_add = {"existance": True, "id": None, "response": None}
        return df, df_add

    mock_obj.side_effect = mock_process_data
    return mock_obj


@pytest.fixture(params=[0, 1, 2])
def close_entity_true(request) -> pd.DataFrame:
    """Fixture providing expected close entity DataFrames for different radius values."""
    param = request.param
    if param in {0, 1, 2}:  # r=2.0, r=12.0, or r=0.5 (search expands to find a result)
        return pd.DataFrame(
            {
                "id": [123],
                "title": ["Test Location"],
                "easting": [2.0],
                "northing": [50.0],
            }
        )
    return pd.DataFrame()


@pytest.fixture
def mock_requests_search_any_entity(mocker: mock.Mock) -> mock.Mock:
    """Mock for search any entity requests."""
    mock_obj = mocker.patch("owi.metadatabase.soil.io.API.process_data")

    def mock_process_data(url_data_type, url_params, output_type):
        radius = float(url_params.get("offset", 2.0))

        # For radius < 1.0 (e.g., 0.5), return no results
        if radius < 1.0:
            df = pd.DataFrame()
            df_add = {"existance": False, "id": None, "response": None}
        else:
            df = pd.DataFrame(
                {
                    "id": [123],
                    "title": ["Test Location"],
                    "easting": [2.0],
                    "northing": [50.0],
                }
            )
            df_add = {"existance": True, "id": 123, "response": None}
        return df, df_add

    mock_obj.side_effect = mock_process_data
    return mock_obj


@pytest.fixture(params=["regular", "single"])
def df_gathered_inp(request) -> pd.DataFrame:
    """Fixture providing input DataFrames for gather_data_entity tests."""
    param = request.param
    if param == "regular":
        return pd.DataFrame(
            {
                "col_1": [11, 21],
                "col_2": [12.5, 22.5],
                "offset [m]": [5.0, 10.0],
                "id": [123, 456],
                "title": ["Test Location 1", "Test Location 2"],
            }
        )
    elif param == "single":
        return pd.DataFrame(
            {
                "col_1": [11],
                "col_2": [12.5],
                "offset [m]": [5.0],
                "id": [123],
                "title": ["Test Location"],
            }
        )
    return pd.DataFrame()


@pytest.fixture(params=["regular", "single"])
def dict_gathered_true(request) -> dict[str, Union[pd.DataFrame, str, float, int]]:
    """Fixture providing expected gathered data dictionaries."""
    param = request.param
    if param == "regular":
        return {
            "data": pd.DataFrame(
                {
                    "col_1": [11, 21],
                    "col_2": [12.5, 22.5],
                    "offset [m]": [5.0, 10.0],
                    "id": [123, 456],
                    "title": ["Test Location 1", "Test Location 2"],
                }
            ),
            "id": 123,
            "title": "Test Location 1",
            "offset [m]": 5.0,
        }
    elif param == "single":
        return {
            "data": pd.DataFrame(
                {
                    "col_1": [11],
                    "col_2": [12.5],
                    "offset [m]": [5.0],
                    "id": [123],
                    "title": ["Test Location"],
                }
            ),
            "id": 123,
            "title": "Test Location",
            "offset [m]": 5.0,
        }
    return {}


@pytest.fixture
def dict_gathered_final_true() -> dict[str, Union[pd.DataFrame, str, float, int]]:
    """Fixture providing expected final gathered data dictionary."""
    return {
        "data": pd.DataFrame(
            {
                "id": [123],
                "title": ["Test Location"],
                "easting": [2.0],
                "northing": [50.0],
                "easting [m]": [428333.5524958731],
                "northing [m]": [5539110.0],
                "offset [m]": [0.0],
            }
        ),
        "id": 123,
        "title": "Test Location",
        "offset [m]": 0.0,
    }
