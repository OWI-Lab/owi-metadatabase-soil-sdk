"""Soil API client for the OWI metadatabase extension.

This module exposes :class:`SoilAPI`, which extends the base
``owi.metadatabase.io.API`` class with soil-specific endpoints.

Examples
--------
>>> from owi.metadatabase.soil import SoilAPI
>>> isinstance(SoilAPI(token="dummy"), SoilAPI)
True
"""

# mypy: ignore-errors

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Union, cast

import numpy as np
import pandas as pd
import requests
from owi.metadatabase._utils.exceptions import InvalidParameterError
from owi.metadatabase.io import API

from owi.metadatabase.soil.processing import SoilDataProcessor

if TYPE_CHECKING:
    from groundhog.siteinvestigation.insitutests.pcpt_processing import PCPTProcessing

SoilApiResult = dict[str, Union[pd.DataFrame, bool, int, float, str, np.int64, requests.Response, None]]
SoilCptDetailResult = dict[
    str,
    Union[pd.DataFrame, bool, int, np.int64, str, requests.Response, "PCPTProcessing", None],
]
SoilIdResult = Union[int, np.int64, bool, None]


class SoilAPI(API):
    """HTTP client for the soil endpoints of the OWI metadatabase.

    Examples
    --------
    >>> api = SoilAPI(token="dummy")
    >>> api.api_root.endswith('/soildata/')
    True
    """

    def __init__(
        self,
        api_subdir: str = "/soildata/",
        **kwargs,
    ) -> None:
        """Initialize the soil API client.

        Parameters
        ----------
        api_subdir : str, default="/soildata/"
            API sub-path appended to the base root.
        **kwargs
            Forwarded to :class:`owi.metadatabase.io.API`.

        Examples
        --------
        >>> api = SoilAPI(api_subdir='/soildata/', token='dummy')
        >>> '/soildata/' in api.api_root
        True
        """
        super().__init__(**kwargs)
        self.api_root = self.api_root + api_subdir

    def get_proximity_entities_2d(
        self, api_url: str, latitude: float, longitude: float, radius: float, **kwargs
    ) -> SoilApiResult:
        """Find entities within a 2D radius around a geographic point.

        Parameters
        ----------
        api_url : str
            Endpoint suffix for the proximity query.
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float
            Search radius in kilometers.
        **kwargs
            Additional query filters.

        Returns
        -------
        SoilApiResult
            Dictionary with ``data`` and ``exists`` keys.

        Raises
        ------
        InvalidParameterError
            If invalid query values are provided.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> with patch.object(SoilAPI, 'process_data', return_value=(pd.DataFrame({'id': [1]}), {'existance': True})):
        ...     out = api.get_proximity_entities_2d('testlocationproximity', 50.0, 2.0, 1.0)
        >>> out['exists']
        True
        """
        geosearch_params = {"latitude": latitude, "longitude": longitude, "offset": radius}
        url_params = {**geosearch_params, **kwargs}
        url_data_type = api_url
        output_type = "list"
        try:
            df, df_add = self.process_data(url_data_type, url_params, output_type)
        except ValueError as err:
            raise InvalidParameterError(str(err)) from err
        for col in df.columns:
            dtype = df[col].dtype
            if isinstance(dtype, pd.Int64Dtype):
                df[col] = df[col].astype("int64")
            elif isinstance(dtype, pd.Float64Dtype):
                df[col] = df[col].astype("float64")
            elif isinstance(dtype, pd.BooleanDtype):
                df[col] = df[col].astype("bool")
            elif isinstance(dtype, pd.StringDtype):
                df[col] = df[col].astype(object)
        return {"data": df, "exists": df_add["existance"]}

    def _search_any_entity(
        self,
        api_url: str,
        radius_init: float,
        url_params: dict[str, Union[str, float, int, Sequence[Union[str, float, int]], None]],
        radius_max: float = 500.0,
    ) -> pd.DataFrame:
        """Search by expanding radius until at least one entity is found.

        Parameters
        ----------
        api_url : str
            Endpoint suffix for the proximity query.
        radius_init : float
            Initial radius in kilometers.
        url_params : dict
            Query parameters; an ``offset`` parameter is injected internally.
        radius_max : float, default=500.0
            Maximum allowed radius in kilometers.

        Returns
        -------
        pandas.DataFrame
            First non-empty query result.

        Raises
        ------
        ValueError
            If no entity is found up to ``radius_max``.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> responses = [(pd.DataFrame(), {'existance': False}), (pd.DataFrame({'id': [1]}), {'existance': True})]
        >>> with patch.object(SoilAPI, 'process_data', side_effect=responses):
        ...     df = api._search_any_entity('test', 0.5, {'latitude': 50.0, 'longitude': 2.0})
        >>> int(df['id'].iloc[0])
        1
        """
        radius = radius_init
        while True:
            url_params["offset"] = str(radius)
            url_data_type = api_url
            output_type = "list"
            df, df_add = self.process_data(url_data_type, url_params, output_type)
            if df_add["existance"]:
                break
            radius *= 2
            warnings.warn(f"Expanding search radius to {radius: .1f}km", stacklevel=2)
            if radius > radius_max:
                raise ValueError(f"No locations found within {radius_max}km radius. Check your input information.")
        return df

    def get_closest_entity_2d(
        self,
        api_url: str,
        latitude: float,
        longitude: float,
        radius_init: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> SoilApiResult:
        """Return the nearest entity in 2D for a given endpoint.

        Parameters
        ----------
        api_url : str
            Endpoint suffix for the proximity query.
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius_init : float, default=1.0
            Initial search radius in kilometers.
        target_srid : str, default="25831"
            SRID used for metric distance calculations.
        **kwargs
            Additional query filters.

        Returns
        -------
        SoilApiResult
            Dictionary with nearest entity metadata and full data table.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> sample = pd.DataFrame({'id': [1], 'title': ['A'], 'easting': [2.0], 'northing': [50.0]})
        >>> with patch.object(SoilAPI, '_search_any_entity', return_value=sample):
        ...     out = api.get_closest_entity_2d('test', 50.0, 2.0)
        >>> int(out['id'])
        1
        """
        geosearch_params = {"latitude": latitude, "longitude": longitude}
        url_params = {**geosearch_params, **kwargs}
        df = self._search_any_entity(api_url, radius_init, url_params)
        df, point_east, point_north = SoilDataProcessor.transform_coord(df, longitude, latitude, target_srid)
        df["offset [m]"] = np.sqrt((df["easting [m]"] - point_east) ** 2 + (df["northing [m]"] - point_north) ** 2)
        return cast(SoilApiResult, SoilDataProcessor.gather_data_entity(df))

    def get_closest_entity_3d(
        self,
        api_url: str,
        latitude: float,
        longitude: float,
        depth: float,
        radius_init: float = 1.0,
        target_srid: str = "25831",
        sampletest: bool = True,
        **kwargs,
    ) -> SoilApiResult:
        """Return the nearest entity in 3D for a given endpoint.

        Parameters
        ----------
        api_url : str
            Endpoint suffix for the proximity query.
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        depth : float
            Reference depth in meters below seabed.
        radius_init : float, default=1.0
            Initial search radius in kilometers.
        target_srid : str, default="25831"
            SRID used for metric distance calculations.
        sampletest : bool, default=True
            Whether rows are sample tests (uses ``depth`` directly) or samples
            (depth inferred from ``top_depth`` and ``bottom_depth``).
        **kwargs
            Additional query filters.

        Returns
        -------
        SoilApiResult
            Dictionary with nearest entity metadata and full data table.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> sample = pd.DataFrame({'id': [1], 'title': ['A'], 'easting': [2.0], 'northing': [50.0], 'depth': [5.0]})
        >>> with patch.object(SoilAPI, '_search_any_entity', return_value=sample):
        ...     out = api.get_closest_entity_3d('test', 50.0, 2.0, depth=5.0)
        >>> int(out['id'])
        1
        """
        geosearch_params = {"latitude": latitude, "longitude": longitude}
        url_params = {**geosearch_params, **kwargs}
        df = self._search_any_entity(api_url, radius_init, url_params)
        df, point_east, point_north = SoilDataProcessor.transform_coord(df, longitude, latitude, target_srid)
        if not sampletest:
            df["depth"] = 0.5 * (df["top_depth"] + df["bottom_depth"])
        df["offset [m]"] = np.sqrt(
            (df["easting [m]"] - point_east) ** 2 + (df["northing [m]"] - point_north) ** 2 + (df["depth"] - depth) ** 2
        )
        return cast(SoilApiResult, SoilDataProcessor.gather_data_entity(df))

    def get_surveycampaigns(self, projectsite: Union[str, None] = None, **kwargs) -> SoilApiResult:
        """Retrieve survey campaigns, optionally filtered by project site.

        Parameters
        ----------
        projectsite : str or None, default=None
            Project site name.
        **kwargs
            Additional query filters.

        Returns
        -------
        SoilApiResult
            Dictionary with ``data`` and ``exists`` keys.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> payload = (
        ...     pd.DataFrame({'title': ['C1']}),
        ...     {'existance': True, 'id': None, 'response': None},
        ... )
        >>> with patch.object(SoilAPI, 'process_data', return_value=payload):
        ...     out = api.get_surveycampaigns(projectsite='Demo')
        >>> out['exists']
        True
        """
        url_params = {"projectsite": projectsite}
        url_params = {**url_params, **kwargs}
        url_data_type = "surveycampaign"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_surveycampaign_detail(
        self,
        projectsite: str,
        campaign: str,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve details for a single survey campaign.

        Parameters
        ----------
        projectsite : str
            Project site name.
        campaign : str
            Survey campaign title.
        **kwargs
            Additional query filters.

        Returns
        -------
        SoilApiResult
            Dictionary with ``id``, ``data`` and ``exists`` keys.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> payload = (pd.DataFrame({'title': ['C1']}), {'existance': True, 'id': 3, 'response': None})
        >>> with patch.object(SoilAPI, 'process_data', return_value=payload):
        ...     out = api.get_surveycampaign_detail('P', 'C1')
        >>> int(out['id'])
        3
        """
        url_params = {"projectsite": projectsite, "campaign": campaign}
        url_params = {**url_params, **kwargs}
        url_data_type = "surveycampaign"
        output_type = "single"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"id": df_add["id"], "data": df, "exists": df_add["existance"]}

    def get_proximity_testlocations(self, latitude: float, longitude: float, radius: float, **kwargs) -> SoilApiResult:
        """Retrieve test locations within a radius around a coordinate.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float
            Radius in kilometers.
        **kwargs
            Additional query filters.

        Returns
        -------
        SoilApiResult
            Dictionary with ``data`` and ``exists`` keys.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> prox = {'data': pd.DataFrame(), 'exists': False}
        >>> with patch.object(SoilAPI, 'get_proximity_entities_2d', return_value=prox):
        ...     out = api.get_proximity_testlocations(50.0, 2.0, 1.0)
        >>> out['exists']
        False
        """
        return self.get_proximity_entities_2d(
            api_url="testlocationproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def get_closest_testlocation(
        self,
        latitude: float,
        longitude: float,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve the closest test location around a coordinate.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float, default=1.0
            Initial search radius in kilometers.
        target_srid : str, default="25831"
            SRID for metric distance calculations.
        **kwargs
            Additional query filters.

        Returns
        -------
        SoilApiResult
            Dictionary including selected ``id``, ``title`` and ``offset [m]``.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> closest = {'data': pd.DataFrame(), 'id': 1, 'title': 'A', 'offset [m]': 0.0}
        >>> with patch.object(SoilAPI, 'get_closest_entity_2d', return_value=closest):
        ...     out = api.get_closest_testlocation(50.0, 2.0)
        >>> out['id']
        1
        """
        return self.get_closest_entity_2d(
            api_url="testlocationproximity",
            latitude=latitude,
            longitude=longitude,
            radius_init=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def get_testlocations(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve geotechnical test locations matching search filters.

        Parameters
        ----------
        projectsite : str or None, default=None
            Project site filter.
        campaign : str or None, default=None
            Survey campaign filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with ``data`` and ``exists`` keys.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> payload = (pd.DataFrame({'title': ['L1']}), {'existance': True, 'id': None, 'response': None})
        >>> with patch.object(SoilAPI, 'process_data', return_value=payload):
        ...     out = api.get_testlocations(projectsite='P')
        >>> out['exists']
        True
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "testlocation"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_testlocation_detail(
        self,
        location: str,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve details for a specific geotechnical test location.

        Parameters
        ----------
        location : str
            Location title.
        projectsite : str or None, default=None
            Project site filter.
        campaign : str or None, default=None
            Campaign filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with ``id``, ``data`` and ``exists`` keys.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> payload = (pd.DataFrame({'title': ['L1']}), {'existance': True, 'id': 10, 'response': None})
        >>> with patch.object(SoilAPI, 'process_data', return_value=payload):
        ...     out = api.get_testlocation_detail('L1')
        >>> int(out['id'])
        10
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "testlocation"
        output_type = "single"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"id": df_add["id"], "data": df, "exists": df_add["existance"]}

    def testlocation_exists(
        self,
        location: str,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> SoilIdResult:
        """Check whether a test location exists and return its id.

        Parameters
        ----------
        location : str
            Location title.
        projectsite : str or None, default=None
            Project site filter.
        campaign : str or None, default=None
            Campaign filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Location id when found; otherwise ``False``.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> api = SoilAPI(token="dummy")
        >>> payload = (None, {'existance': True, 'id': 4, 'response': None})
        >>> with patch.object(SoilAPI, 'process_data', return_value=payload):
        ...     out = api.testlocation_exists('L1')
        >>> int(out)
        4
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "testlocation"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_insitutest_types(self, **kwargs):
        """Retrieve available in-situ test types.

        Parameters
        ----------
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with available test type rows and existence flag.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> payload = (
        ...     pd.DataFrame({'title': ['PCPT']}),
        ...     {'existance': True, 'id': None, 'response': None},
        ... )
        >>> with patch.object(SoilAPI, 'process_data', return_value=payload):
        ...     out = api.get_insitutest_types()
        >>> out['exists']
        True
        """
        url_data_type = "insitutesttype"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, {}, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def insitutest_type_exists(self, testtype: str, **kwargs) -> SoilIdResult:
        """Check whether an in-situ test type exists and return its id.

        Parameters
        ----------
        testtype : str
            In-situ test type title.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Type id when found; otherwise ``False``.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> api = SoilAPI(token="dummy")
        >>> payload = (None, {'existance': False, 'id': None, 'response': None})
        >>> with patch.object(SoilAPI, 'process_data', return_value=payload):
        ...     out = api.insitutest_type_exists('PCPT')
        >>> out
        False
        """
        url_params = {"testtype": testtype}
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutesttype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_insitutests(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        insitutest: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve in-situ tests matching provided filters.

        Parameters
        ----------
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Test location filter.
        testtype : str or None, default=None
            In-situ test type filter.
        insitutest : str or None, default=None
            In-situ test title filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with in-situ test summary rows and existence flag.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> payload = (pd.DataFrame({'title': ['CPT-1']}), {'existance': True, 'id': None, 'response': None})
        >>> with patch.object(SoilAPI, 'process_data', return_value=payload):
        ...     out = api.get_insitutests(projectsite='P')
        >>> out['exists']
        True
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "testtype": testtype,
            "insitutest": insitutest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutestsummary"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_proximity_insitutests(self, latitude: float, longitude: float, radius: float, **kwargs) -> SoilApiResult:
        """Retrieve in-situ tests in a radius around a coordinate.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float
            Radius in kilometers.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with in-situ test rows and existence flag.

        Examples
        --------
        >>> from unittest.mock import patch
        >>> import pandas as pd
        >>> api = SoilAPI(token="dummy")
        >>> prox = {'data': pd.DataFrame(), 'exists': False}
        >>> with patch.object(SoilAPI, 'get_proximity_entities_2d', return_value=prox):
        ...     out = api.get_proximity_insitutests(50.0, 2.0, 1.0)
        >>> out['exists']
        False
        """
        return self.get_proximity_entities_2d(
            api_url="insitutestproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def get_closest_insitutest(
        self,
        latitude: float,
        longitude: float,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ):
        """Return the closest in-situ test for a coordinate.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float, default=1.0
            Initial search radius in kilometers.
        target_srid : str, default="25831"
            SRID for metric distance calculations.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with selected test metadata and candidate data.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_closest_insitutest(50.0, 2.0)  # doctest: +SKIP
        """
        return self.get_closest_entity_2d(
            api_url="insitutestproximity",
            latitude=latitude,
            longitude=longitude,
            radius_init=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def get_closest_insitutest_byname(
        self,
        projectsite: str,
        location: str,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> SoilApiResult:
        """Return the closest in-situ test around a named location.

        Parameters
        ----------
        projectsite : str
            Project site name.
        location : str
            Location title.
        radius : float, default=1.0
            Initial search radius in kilometers.
        target_srid : str, default="25831"
            SRID for metric distance calculations.
        **kwargs
            Additional test filters.

        Returns
        -------
        SoilApiResult
            Dictionary with selected in-situ test metadata.

        Raises
        ------
        ValueError
            If the location is not found.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_closest_insitutest_byname('Nobelwind', 'CPT-7C')  # doctest: +SKIP
        """
        # First verify the location exists
        location_id = self.testlocation_exists(projectsite=projectsite, location=location)

        if not location_id:
            raise ValueError(f"Location '{location}' not found in project '{projectsite}'")

        # Get location details to obtain coordinates
        location_details = self.get_testlocation_detail(projectsite=projectsite, location=location)

        # Extract coordinates from location data
        location_data = cast(pd.DataFrame, location_details["data"])
        latitude = location_data["latitude"].iloc[0]
        longitude = location_data["longitude"].iloc[0]

        # Use existing method to find closest in-situ test
        return self.get_closest_insitutest(
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def get_closest_soilprofile_byname(
        self,
        projectsite: str,
        location: str,
        radius: float = 1.0,
        target_srid: str = "25831",
        retrieve_details: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> SoilApiResult:
        """Return the closest soil profile around a named location.

        Parameters
        ----------
        projectsite : str
            Project site name.
        location : str
            Location title.
        radius : float, default=1.0
            Initial search radius in kilometers.
        target_srid : str, default="25831"
            SRID for metric distance calculations.
        retrieve_details : bool, default=True
            Return full profile details when ``True``.
        verbose : bool, default=True
            Print selected profile info.
        **kwargs
            Additional profile filters.

        Returns
        -------
        SoilApiResult
            Closest profile summary or full detail dictionary.

        Raises
        ------
        ValueError
            If the location is not found.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_closest_soilprofile_byname('Nobelwind', 'CPT-7C')  # doctest: +SKIP
        """
        # First verify the location exists
        location_id = self.testlocation_exists(projectsite=projectsite, location=location)

        if not location_id:
            raise ValueError(f"Location '{location}' not found in project '{projectsite}'")

        # Get location details to obtain coordinates
        location_details = self.get_testlocation_detail(projectsite=projectsite, location=location)

        # Extract coordinates from location data
        location_data = cast(pd.DataFrame, location_details["data"])
        latitude = location_data["latitude"].iloc[0]
        longitude = location_data["longitude"].iloc[0]

        # Use existing method to find closest soil profile
        closest_profile = self.get_closest_soilprofile(
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            target_srid=target_srid,
            **kwargs,
        )

        if verbose:
            print(f"Soil profile {closest_profile['title']} found at {closest_profile['offset [m]']:.1f}m offset")

        if retrieve_details:
            soilprofile_title = cast(str, closest_profile["title"])
            return self.get_soilprofile_detail(
                projectsite=projectsite,
                location=location,
                soilprofile=soilprofile_title,
                **kwargs,
            )

        return closest_profile

    def get_insitutest_detail(
        self,
        insitutest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        combine: bool = False,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve detail records for a single in-situ test.

        Parameters
        ----------
        insitutest : str
            In-situ test title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        testtype : str or None, default=None
            Test type filter.
        combine : bool, default=False
            Merge raw and processed tables on depth when ``True``.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with summary, raw/processed/condition tables and status.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_insitutest_detail('CPT-1')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "testtype": testtype,
            "insitutest": insitutest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutestsummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "insitutestdetail"
        df_detail, df_add_detail = self.process_data(url_data_type, url_params, output_type)
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = SoilDataProcessor.process_insitutest_dfs(df_detail, cols)
        df_raw = SoilDataProcessor.combine_dfs(dfs) if combine else dfs["rawdata"]
        return {
            "id": df_add_detail["id"],
            "insitutestsummary": df_sum,
            "rawdata": df_raw,
            "processeddata": dfs["processeddata"],
            "conditions": dfs["conditions"],
            "response": df_add_detail["response"],
            "exists": df_add_sum["existance"],
        }

    def get_cpttest_detail(
        self,
        insitutest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        combine: bool = False,
        cpt: bool = True,
        **kwargs,
    ) -> SoilCptDetailResult:
        """Retrieve CPT test details and optionally build a CPT object.

        Parameters
        ----------
        insitutest : str
            CPT in-situ test title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        testtype : str or None, default=None
            Test type filter.
        combine : bool, default=False
            Merge raw and processed tables on depth when ``True``.
        cpt : bool, default=True
            Build and include a ``PCPTProcessing`` object.
        **kwargs
            Forwarded to CPT object loading.

        Returns
        -------
        SoilCptDetailResult
            Detail dictionary with optional ``cpt`` object.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_cpttest_detail('CPT-1')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "testtype": testtype,
            "insitutest": insitutest,
        }
        url_data_type = "insitutestsummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "insitutestdetail"
        df_detail, df_add_detail = self.process_data(url_data_type, url_params, output_type)
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = SoilDataProcessor.process_insitutest_dfs(df_detail, cols)
        df_raw = SoilDataProcessor.combine_dfs(dfs) if combine else dfs["rawdata"]
        dict_ = {
            "id": df_add_detail["id"],
            "insitutestsummary": df_sum,
            "rawdata": df_raw,
            "processeddata": dfs["processeddata"],
            "conditions": dfs["conditions"],
            "response": df_add_detail["response"],
            "exists": df_add_sum["existance"],
        }
        if cpt:
            cpt_ = SoilDataProcessor.process_cpt(df_sum, df_raw, **kwargs)
            dict_["cpt"] = cpt_
            return dict_
        return dict_

    def insitutest_exists(
        self,
        insitutest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        **kwargs,
    ) -> SoilIdResult:
        """Check whether an in-situ test exists and return its id.

        Parameters
        ----------
        insitutest : str
            In-situ test title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        testtype : str or None, default=None
            Test type filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Test id when found; otherwise ``False``.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.insitutest_exists('CPT-1')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "testtype": testtype,
            "insitutest": insitutest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "insitutestdetail"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_soilprofiles(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        soilprofile: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve soil profile summaries matching search filters.

        Parameters
        ----------
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        soilprofile : str or None, default=None
            Soil profile title filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Summary table and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_soilprofiles(projectsite='Nobelwind')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "soilprofile": soilprofile,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilprofilesummary"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_proximity_soilprofiles(self, latitude: float, longitude: float, radius: float, **kwargs) -> SoilApiResult:
        """Retrieve soil profiles within a radius around a coordinate.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float
            Radius in kilometers.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Profile summary table and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_proximity_soilprofiles(50.0, 2.0, 1.0)  # doctest: +SKIP
        """
        return self.get_proximity_entities_2d(
            api_url="soilprofileproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def get_closest_soilprofile(
        self,
        latitude: float,
        longitude: float,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> SoilApiResult:
        """Return the closest soil profile for a coordinate.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float, default=1.0
            Initial search radius in kilometers.
        target_srid : str, default="25831"
            SRID for metric distance calculations.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with selected profile metadata.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_closest_soilprofile(50.0, 2.0)  # doctest: +SKIP
        """
        return self.get_closest_entity_2d(
            api_url="soilprofileproximity",
            latitude=latitude,
            longitude=longitude,
            radius_init=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def get_soilprofile_detail(
        self,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        soilprofile: Union[str, None] = None,
        convert_to_profile: bool = True,
        profile_title: Union[str, None] = None,
        drop_info_cols: bool = True,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve soil profile details and optional Groundhog conversion.

        Parameters
        ----------
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        soilprofile : str or None, default=None
            Soil profile title filter.
        convert_to_profile : bool, default=True
            Convert detail rows to ``SoilProfile`` object.
        profile_title : str or None, default=None
            Custom title for converted profile.
        drop_info_cols : bool, default=True
            Drop informational columns before conversion.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Detail dictionary with optional converted profile object.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_soilprofile_detail(projectsite='Nobelwind')  # doctest: +SKIP
        """
        # TODO: Ensure that an option for retrieving soilprofiles in mLAT is also available
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "soilprofile": soilprofile,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilprofilesummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "soilprofiledetail"
        df_detail, df_add_detail = self.process_data(url_data_type, url_params, output_type)
        dict_ = {
            "id": df_add_detail["id"],
            "soilprofilesummary": df_sum,
            "response": df_add_detail["response"],
            "exists": df_add_sum["existance"],
        }
        if convert_to_profile:
            dsp = SoilDataProcessor.convert_to_profile(df_sum, df_detail, profile_title, drop_info_cols)
            dict_["soilprofile"] = dsp
            return dict_
        return dict_

    def soilprofile_exists(
        self,
        soilprofile: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        **kwargs,
    ) -> SoilIdResult:
        """Check whether a soil profile exists and return its id.

        Parameters
        ----------
        soilprofile : str
            Soil profile title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Profile id when found; otherwise ``False``.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.soilprofile_exists('Borehole log')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "location": location,
            "soilprofile": soilprofile,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilprofiledetail"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def soiltype_exists(self, soiltype: str, **kwargs) -> SoilIdResult:
        """Check whether a soil type exists and return its id.

        Parameters
        ----------
        soiltype : str
            Soil type name.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Soil type id when found; otherwise ``False``.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.soiltype_exists('SAND')  # doctest: +SKIP
        """
        url_params = {"soiltype": soiltype}
        url_params = {**url_params, **kwargs}
        url_data_type = "soiltype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def soilunit_exists(
        self,
        soilunit: str,
        projectsite: Union[str, None] = None,
        soiltype: Union[str, None] = None,
        **kwargs,
    ) -> SoilIdResult:
        """Check whether a soil unit exists and return its id.

        Parameters
        ----------
        soilunit : str
            Soil unit name.
        projectsite : str or None, default=None
            Project site filter.
        soiltype : str or None, default=None
            Soil type filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Soil unit id when found; otherwise ``False``.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.soilunit_exists('Asse sand-clay')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "soiltype": soiltype,
            "soilunit": soilunit,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilunit"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_soilunits(
        self,
        projectsite: Union[str, None] = None,
        soiltype: Union[str, None] = None,
        soilunit: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve soil units matching search filters.

        Parameters
        ----------
        projectsite : str or None, default=None
            Project site filter.
        soiltype : str or None, default=None
            Soil type filter.
        soilunit : str or None, default=None
            Soil unit filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Soil unit rows and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_soilunits(projectsite='HKN')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "soiltype": soiltype,
            "soilunit": soilunit,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soilunit"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_batchlabtest_types(self, **kwargs) -> SoilApiResult:
        """Retrieve available batch lab test types.

        Parameters
        ----------
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Batch test type rows and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_batchlabtest_types()  # doctest: +SKIP
        """
        url_data_type = "batchlabtesttype"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, kwargs, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_batchlabtests(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        batchlabtest: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve batch lab test summaries matching search filters.

        Parameters
        ----------
        projectsite : str or None, default=None
            Project site filter.
        campaign : str or None, default=None
            Campaign filter.
        location : str or None, default=None
            Location filter.
        testtype : str or None, default=None
            Test type filter.
        batchlabtest : str or None, default=None
            Batch lab test title filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Batch test summary rows and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_batchlabtests(projectsite='Nobelwind')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "testtype": testtype,
            "batchlabtest": batchlabtest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "batchlabtestsummary"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def batchlabtesttype_exists(self, batchlabtesttype: str, **kwargs) -> SoilIdResult:
        """Check whether a batch lab test type exists and return its id.

        Parameters
        ----------
        batchlabtesttype : str
            Batch lab test type title.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Type id when found; otherwise ``False``.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.batchlabtesttype_exists('Atterberg')  # doctest: +SKIP
        """
        url_params = {"testtype": batchlabtesttype}
        url_params = {**url_params, **kwargs}
        url_data_type = "batchlabtesttype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_proximity_batchlabtests(self, latitude: float, longitude: float, radius: float, **kwargs) -> SoilApiResult:
        """Retrieve batch lab tests within a radius around a coordinate.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float
            Radius in kilometers.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Batch test summary rows and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_proximity_batchlabtests(50.0, 2.0, 1.0)  # doctest: +SKIP
        """
        return self.get_proximity_entities_2d(
            api_url="batchlabtestproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def get_closest_batchlabtest(
        self,
        latitude: float,
        longitude: float,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ):
        """Return the closest batch lab test for a coordinate.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float, default=1.0
            Initial search radius in kilometers.
        target_srid : str, default="25831"
            SRID for metric distance calculations.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with selected test metadata.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_closest_batchlabtest(50.0, 2.0)  # doctest: +SKIP
        """
        return self.get_closest_entity_2d(
            api_url="batchlabtestproximity",
            latitude=latitude,
            longitude=longitude,
            radius_init=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def get_batchlabtest_detail(
        self,
        batchlabtest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve detailed data for one batch lab test.

        Parameters
        ----------
        batchlabtest : str
            Batch lab test title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        testtype : str or None, default=None
            Test type filter.
        campaign : str or None, default=None
            Campaign filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Detail dictionary with summary/raw/processed/conditions tables.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_batchlabtest_detail('Atterberg-1')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "testtype": testtype,
            "batchlabtest": batchlabtest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "batchlabtestsummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "batchlabtestdetail"
        df_detail, df_add_detail = self.process_data(url_data_type, url_params, output_type)
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = SoilDataProcessor.process_insitutest_dfs(df_detail, cols)
        return {
            "id": df_add_detail["id"],
            "summary": df_sum,
            "response": df_add_detail["response"],
            "rawdata": dfs["rawdata"],
            "processeddata": dfs["processeddata"],
            "conditions": dfs["conditions"],
            "exists": df_add_sum["existance"],
        }

    def batchlabtest_exists(
        self,
        batchlabtest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> SoilIdResult:
        """Check whether a batch lab test exists and return its id.

        Parameters
        ----------
        batchlabtest : str
            Batch lab test title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        testtype : str or None, default=None
            Test type filter.
        campaign : str or None, default=None
            Campaign filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Test id when found; otherwise ``False``.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.batchlabtest_exists('Atterberg-1')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "testtype": testtype,
            "batchlabtest": batchlabtest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "batchlabtestdetail"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def geotechnicalsampletype_exists(self, sampletype: str, **kwargs) -> SoilIdResult:
        """Check whether a geotechnical sample type exists.

        Parameters
        ----------
        sampletype : str
            Sample type title.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Sample type id when found; otherwise ``False``.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.geotechnicalsampletype_exists('Core')  # doctest: +SKIP
        """
        url_params = {"sampletype": sampletype}
        url_params = {**url_params, **kwargs}
        url_data_type = "geotechnicalsampletype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_geotechnicalsamples(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        sampletype: Union[str, None] = None,
        sample: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve geotechnical sample summaries matching filters.

        Parameters
        ----------
        projectsite : str or None, default=None
            Project site filter.
        campaign : str or None, default=None
            Campaign filter.
        location : str or None, default=None
            Location filter.
        sampletype : str or None, default=None
            Sample type filter.
        sample : str or None, default=None
            Sample title filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Sample rows and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_geotechnicalsamples(projectsite='Nobelwind')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sampletype": sampletype,
            "sample": sample,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "geotechnicalsample"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_proximity_geotechnicalsamples(
        self, latitude: float, longitude: float, radius: float, **kwargs
    ) -> SoilApiResult:
        """Retrieve geotechnical samples around a coordinate.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float
            Radius in kilometers.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Sample rows and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_proximity_geotechnicalsamples(50.0, 2.0, 1.0)  # doctest: +SKIP
        """
        return self.get_proximity_entities_2d(
            api_url="geotechnicalsampleproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def get_closest_geotechnicalsample(
        self,
        latitude: float,
        longitude: float,
        depth: float,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> SoilApiResult:
        """Return the closest geotechnical sample for a 3D query point.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        depth : float
            Query depth in meters below seabed.
        radius : float, default=1.0
            Initial search radius in kilometers.
        target_srid : str, default="25831"
            SRID for metric distance calculations.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with selected sample metadata.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_closest_geotechnicalsample(50.0, 2.0, 5.0)  # doctest: +SKIP
        """
        return self.get_closest_entity_3d(
            api_url="geotechnicalsampleproximity",
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            radius_init=radius,
            target_srid=target_srid,
            sampletest=False,
            **kwargs,
        )

    def get_geotechnicalsample_detail(
        self,
        sample: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        sampletype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve details for one geotechnical sample.

        Parameters
        ----------
        sample : str
            Sample title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        sampletype : str or None, default=None
            Sample type filter.
        campaign : str or None, default=None
            Campaign filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Sample detail dictionary with response and existence status.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_geotechnicalsample_detail('S-1')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sampletype": sampletype,
            "sample": sample,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "geotechnicalsample"
        output_type = "single"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {
            "id": df_add["id"],
            "data": df,
            "response": df_add["response"],
            "exists": df_add["existance"],
        }

    def geotechnicalsample_exists(
        self,
        sample: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        sampletype: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> SoilIdResult:
        """Check whether a geotechnical sample exists.

        Parameters
        ----------
        sample : str
            Sample title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        sampletype : str or None, default=None
            Sample type filter.
        campaign : str or None, default=None
            Campaign filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Sample id when found; otherwise ``False``.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.geotechnicalsample_exists('S-1')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sampletype": sampletype,
            "sample": sample,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "geotechnicalsample"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_sampletests(
        self,
        projectsite: Union[str, None] = None,
        campaign: Union[str, None] = None,
        location: Union[str, None] = None,
        sample: Union[str, None] = None,
        testtype: Union[str, None] = None,
        sampletest: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve sample test summaries matching filters.

        Parameters
        ----------
        projectsite : str or None, default=None
            Project site filter.
        campaign : str or None, default=None
            Campaign filter.
        location : str or None, default=None
            Location filter.
        sample : str or None, default=None
            Sample title filter.
        testtype : str or None, default=None
            Sample test type filter.
        sampletest : str or None, default=None
            Sample test title filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Sample test summary rows and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_sampletests(projectsite='Nobelwind')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sample": sample,
            "testtype": testtype,
            "sampletest": sampletest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "sampletestsummary"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, url_params, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_proximity_sampletests(self, latitude: float, longitude: float, radius: float, **kwargs) -> SoilApiResult:
        """Retrieve sample tests around a coordinate.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        radius : float
            Radius in kilometers.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Sample test summary rows and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_proximity_sampletests(50.0, 2.0, 1.0)  # doctest: +SKIP
        """
        return self.get_proximity_entities_2d(
            api_url="sampletestproximity",
            latitude=latitude,
            longitude=longitude,
            radius=radius,
            **kwargs,
        )

    def get_closest_sampletest(
        self,
        latitude: float,
        longitude: float,
        depth: float,
        radius: float = 1.0,
        target_srid: str = "25831",
        **kwargs,
    ) -> SoilApiResult:
        """Return the closest sample test for a 3D query point.

        Parameters
        ----------
        latitude : float
            Latitude in decimal degrees.
        longitude : float
            Longitude in decimal degrees.
        depth : float
            Query depth in meters below seabed.
        radius : float, default=1.0
            Initial search radius in kilometers.
        target_srid : str, default="25831"
            SRID for metric distance calculations.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Dictionary with selected sample-test metadata.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_closest_sampletest(50.0, 2.0, 5.0)  # doctest: +SKIP
        """
        return self.get_closest_entity_3d(
            api_url="sampletestproximity",
            latitude=latitude,
            longitude=longitude,
            depth=depth,
            radius_init=radius,
            target_srid=target_srid,
            **kwargs,
        )

    def sampletesttype_exists(self, sampletesttype: str, **kwargs) -> SoilIdResult:
        """Check whether a sample test type exists.

        Parameters
        ----------
        sampletesttype : str
            Sample test type title.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Type id when found; otherwise ``False``.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.sampletesttype_exists('UCS')  # doctest: +SKIP
        """
        url_params = {"testtype": sampletesttype}
        url_params = {**url_params, **kwargs}
        url_data_type = "sampletesttype"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_sampletesttypes(self, **kwargs) -> SoilApiResult:
        """Retrieve available sample test types.

        Parameters
        ----------
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Sample test type rows and existence flag.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_sampletesttypes()  # doctest: +SKIP
        """
        url_data_type = "sampletesttype"
        output_type = "list"
        df, df_add = self.process_data(url_data_type, kwargs, output_type)
        return {"data": df, "exists": df_add["existance"]}

    def get_sampletest_detail(
        self,
        sampletest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        sample: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> SoilApiResult:
        """Retrieve detailed data for one sample test.

        Parameters
        ----------
        sampletest : str
            Sample test title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        testtype : str or None, default=None
            Test type filter.
        sample : str or None, default=None
            Sample filter.
        campaign : str or None, default=None
            Campaign filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        SoilApiResult
            Detail dictionary with summary/raw/processed/conditions tables.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_sampletest_detail('UCS-1')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sample": sample,
            "testtype": testtype,
            "sampletest": sampletest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "sampletestsummary"
        output_type = "single"
        df_sum, df_add_sum = self.process_data(url_data_type, url_params, output_type)
        url_data_type = "sampletestdetail"
        df_detail, df_add_detail = self.process_data(url_data_type, url_params, output_type)
        cols = ["rawdata", "processeddata", "conditions"]
        dfs = SoilDataProcessor.process_insitutest_dfs(df_detail, cols)
        return {
            "id": df_add_detail["id"],
            "summary": df_sum,
            "rawdata": dfs["rawdata"],
            "processeddata": dfs["processeddata"],
            "conditions": dfs["conditions"],
            "response": df_add_detail["response"],
            "exists": df_add_sum["existance"],
        }

    def sampletest_exists(
        self,
        sampletest: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        testtype: Union[str, None] = None,
        sample: Union[str, None] = None,
        campaign: Union[str, None] = None,
        **kwargs,
    ) -> SoilIdResult:
        """Check whether a sample test exists and return its id.

        Parameters
        ----------
        sampletest : str
            Sample test title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        testtype : str or None, default=None
            Test type filter.
        sample : str or None, default=None
            Sample filter.
        campaign : str or None, default=None
            Campaign filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        int | numpy.int64 | bool | None
            Sample test id when found; otherwise ``False``.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.sampletest_exists('UCS-1')  # doctest: +SKIP
        """
        url_params = {
            "projectsite": projectsite,
            "campaign": campaign,
            "location": location,
            "sample": sample,
            "testtype": testtype,
            "sampletest": sampletest,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "sampletestdetail"
        output_type = "single"
        _, df_add = self.process_data(url_data_type, url_params, output_type)
        return df_add["id"] if df_add["existance"] else False

    def get_soilunit_depthranges(
        self,
        soilunit: str,
        projectsite: Union[str, None] = None,
        location: Union[str, None] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Retrieve depth intervals where a soil unit occurs.

        Parameters
        ----------
        soilunit : str
            Soil unit title.
        projectsite : str or None, default=None
            Project site filter.
        location : str or None, default=None
            Location filter.
        **kwargs
            Additional queryset filters.

        Returns
        -------
        pandas.DataFrame
            Table of depth intervals for the selected soil unit.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_soilunit_depthranges('Asse sand-clay')  # doctest: +SKIP
        """
        url_params = {
            "soilunit": soilunit,
            "projectsite": projectsite,
            "location": location,
        }
        url_params = {**url_params, **kwargs}
        url_data_type = "soillayer"
        output_type = "list"
        df, _ = self.process_data(url_data_type, url_params, output_type)
        return df

    def get_unit_insitutestdata(self, soilunit: str, depthcol: Union[str, None] = "z [m]", **kwargs) -> pd.DataFrame:
        """Retrieve in-situ test data that fall inside a soil unit.

        Parameters
        ----------
        soilunit : str
            Soil unit name.
        depthcol : str or None, default="z [m]"
            Depth column in detail ``rawdata`` tables.
        **kwargs
            Filters forwarded to in-situ test retrieval methods.

        Returns
        -------
        pandas.DataFrame
            In-situ test data constrained to selected unit depth ranges.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_unit_insitutestdata('Asse sand-clay')  # doctest: +SKIP
        """
        return self._process_data_units(
            soilunit,
            self.get_insitutests,
            self.get_insitutest_detail,
            depthcol=depthcol,
            **kwargs,
        )

    def get_unit_batchlabtestdata(self, soilunit: str, depthcol: Union[str, None] = "z [m]", **kwargs) -> pd.DataFrame:
        """Retrieve batch lab test data that fall inside a soil unit.

        Parameters
        ----------
        soilunit : str
            Soil unit name.
        depthcol : str or None, default="z [m]"
            Depth column in detail ``rawdata`` tables.
        **kwargs
            Filters forwarded to batch lab test retrieval methods.

        Returns
        -------
        pandas.DataFrame
            Batch test data constrained to selected unit depth ranges.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_unit_batchlabtestdata('Asse sand-clay')  # doctest: +SKIP
        """
        return self._process_data_units(
            soilunit,
            self.get_batchlabtests,
            self.get_batchlabtest_detail,
            depthcol=depthcol,
            **kwargs,
        )

    def get_unit_sampletests(self, soilunit: str, **kwargs) -> pd.DataFrame:
        """Retrieve sample test metadata that fall inside a soil unit.

        Parameters
        ----------
        soilunit : str
            Soil unit name.
        **kwargs
            Filters forwarded to sample test retrieval methods.

        Returns
        -------
        pandas.DataFrame
            Sample test metadata constrained to selected unit depth ranges.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_unit_sampletests('Asse sand-clay')  # doctest: +SKIP
        """
        return self._process_data_units(soilunit, self.get_sampletests, **kwargs)

    def get_soilprofile_profile(
        self, lat1: float, lon1: float, lat2: float, lon2: float, band: float = 1000
    ) -> pd.DataFrame:
        """Retrieve soil profiles along a line segment corridor.

        Parameters
        ----------
        lat1 : float
            Start latitude.
        lon1 : float
            Start longitude.
        lat2 : float
            End latitude.
        lon2 : float
            End longitude.
        band : float, default=1000
            Corridor half-width in meters.

        Returns
        -------
        pandas.DataFrame
            Soil profile summary rows along the profile.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_soilprofile_profile(50.0, 2.0, 50.1, 2.1)  # doctest: +SKIP
        """
        url_params = {
            "lat1": lat1,
            "lon1": lon1,
            "lat2": lat2,
            "lon2": lon2,
            "offset": band,
        }
        url_data_type = "soilprofileprofile"
        output_type = "list"
        df, _ = self.process_data(url_data_type, url_params, output_type)
        return df

    def get_insitutests_profile(
        self, lat1: float, lon1: float, lat2: float, lon2: float, band: float = 1000
    ) -> pd.DataFrame:
        """Retrieve in-situ tests along a line segment corridor.

        Parameters
        ----------
        lat1 : float
            Start latitude.
        lon1 : float
            Start longitude.
        lat2 : float
            End latitude.
        lon2 : float
            End longitude.
        band : float, default=1000
            Corridor half-width in meters.

        Returns
        -------
        pandas.DataFrame
            In-situ test summary rows along the profile.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> api.get_insitutests_profile(50.0, 2.0, 50.1, 2.1)  # doctest: +SKIP
        """
        url_params = {
            "lat1": lat1,
            "lon1": lon1,
            "lat2": lat2,
            "lon2": lon2,
            "offset": band,
        }
        url_data_type = "insitutestprofile"
        output_type = "list"
        df, _ = self.process_data(url_data_type, url_params, output_type)
        return df

    def _process_data_units(
        self,
        soilunit: str,
        func_get: Callable,
        func_get_details: Union[Callable, None] = None,
        depthcol: Union[str, None] = None,
        full: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Collect data inside unit depth windows.

        Parameters
        ----------
        soilunit : str
            Soil unit name.
        func_get : Callable
            Function returning summary rows in a ``data`` key.
        func_get_details : Callable or None, default=None
            Optional function returning detail rows with ``rawdata``.
        depthcol : str or None, default=None
            Depth column name in detail ``rawdata``.
        full : bool, default=True
            Whether to retrieve full detail tables or partial metadata.
        **kwargs
            Forwarded to retrieval functions.

        Returns
        -------
        pandas.DataFrame
            Concatenated data rows contained in selected unit depth ranges.

        Examples
        --------
        >>> api = SoilAPI(token="dummy")
        >>> callable(api._process_data_units)
        True
        """
        selected_depths = self.get_soilunit_depthranges(soilunit=soilunit)
        selected_tests = func_get(**kwargs)["data"]
        all_unit_data = pd.DataFrame()
        for _, row in selected_tests.iterrows():
            unitdata = pd.DataFrame()
            if row["location_name"] in selected_depths["location_name"].unique():
                if full:
                    unitdata = SoilDataProcessor.fulldata_processing(
                        unitdata,
                        row,
                        selected_depths,
                        func_get_details,
                        depthcol,
                        **kwargs,
                    )
                else:
                    unitdata = SoilDataProcessor.partialdata_processing(unitdata, row, selected_depths, selected_tests)
            else:
                print(f"Soil unit not found for {row['location_name']}")
            all_unit_data = pd.concat([all_unit_data, unitdata])
        all_unit_data.reset_index(drop=True, inplace=True)
        return all_unit_data
