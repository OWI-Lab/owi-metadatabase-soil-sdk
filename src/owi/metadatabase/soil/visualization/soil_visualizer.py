"""Visualization helpers for soil data.

The module builds Plotly-based figures from soil profile and CPT data exposed
by :class:`owi.metadatabase.soil.io.SoilAPI`.

Examples
--------
>>> from owi.metadatabase.soil.visualization.soil_visualizer import SoilPlot
>>> SoilPlot.__name__
'SoilPlot'
"""

# mypy: ignore-errors

from typing import Any, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from groundhog.general.soilprofile import plot_fence_diagram
from groundhog.siteinvestigation.insitutests.pcpt_processing import (
    plot_combined_longitudinal_profile,
    plot_longitudinal_profile,
)

from owi.metadatabase.soil.io import SoilAPI
from owi.metadatabase.soil.processing.soil_pp import SoilDataProcessor


class SoilPlot:
    """Create interactive visualizations for soil and CPT datasets.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> plotter = SoilPlot(Mock())
    >>> isinstance(plotter, SoilPlot)
    True
    """

    def __init__(self, soil_api: SoilAPI):
        """Initialize a plotting helper with a soil API instance.

        Parameters
        ----------
        soil_api : SoilAPI
            API client used to fetch soil/CPT data.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> SoilPlot(Mock()).soil_api is not None
        True
        """
        self.soil_api = soil_api

    def plot_soilprofile_fence(
        self,
        soilprofiles_df: pd.DataFrame,
        start: str,
        end: str,
        plotmap: bool = False,
        fillcolordict: Optional[dict[str, str]] = None,
        logwidth: float = 100.0,
        show_annotations: bool = True,
        general_layout: Optional[dict[Any, Any]] = None,
        **kwargs,
    ) -> dict[str, Union[list[pd.DataFrame], go.Figure]]:
        """Create a fence diagram for selected soil profiles.

        Parameters
        ----------
        soilprofiles_df : pandas.DataFrame
            Summary rows for selected soil profiles.
        start : str
            Location name used as start point.
        end : str
            Location name used as end point.
        plotmap : bool, default=False
            If ``True``, include a map panel.
        fillcolordict : dict[str, str] or None, default=None
            Soil-type color mapping.
        logwidth : float, default=100.0
            Width of log traces.
        show_annotations : bool, default=True
            Toggle annotations in the output figure.
        general_layout : dict[Any, Any] or None, default=None
            Extra layout parameters.
        **kwargs
            Forwarded to profile plotting utilities.

        Returns
        -------
        dict[str, list[pandas.DataFrame] | plotly.graph_objs.Figure]
            Dictionary containing loaded profiles and fence diagram.

        Examples
        --------
        >>> import pandas as pd
        >>> from unittest.mock import Mock
        >>> plotter = SoilPlot(Mock())
        >>> sample = pd.DataFrame(
        ...     columns=[
        ...         "projectsite_name",
        ...         "location_name",
        ...         "title",
        ...         "easting",
        ...         "northing",
        ...         "elevation",
        ...     ]
        ... )
        >>> plotter.plot_soilprofile_fence(sample, "A", "B")  # doctest: +SKIP
        """
        if general_layout is None:
            general_layout = {}
        if fillcolordict is None:
            fillcolordict = {
                "SAND": "yellow",
                "CLAY": "brown",
                "SAND/CLAY": "orange",
            }
        soilprofiles = SoilDataProcessor.objects_to_list(
            soilprofiles_df, self.soil_api.get_soilprofile_detail, "soilprofile"
        )
        fence_diagram_1 = plot_fence_diagram(
            profiles=soilprofiles,
            start=start,
            end=end,
            plotmap=plotmap,
            latlon=True,
            fillcolordict=fillcolordict,
            logwidth=logwidth,
            show_annotations=show_annotations,
            general_layout=general_layout,
            **kwargs,
        )
        return {"profiles": soilprofiles, "diagram": fence_diagram_1}

    @staticmethod
    def plot_combined_fence(
        profiles: list[pd.DataFrame],
        cpts: list[pd.DataFrame],
        startpoint: str,
        endpoint: str,
        band: float = 1000.0,
        scale_factor: float = 10.0,
        extend_profile: bool = True,
        show_annotations: bool = True,
        general_layout: Optional[dict[Any, Any]] = None,
        fillcolordict: Optional[dict[str, str]] = None,
        logwidth: float = 100.0,
        opacity: float = 0.5,
        uniformcolor: Union[str, None] = None,
        **kwargs,
    ) -> dict[str, go.Figure]:
        """Create a combined fence for profiles and CPTs.

        Parameters
        ----------
        profiles : list[pandas.DataFrame]
            Georeferenced soil profiles.
        cpts : list[pandas.DataFrame]
            Georeferenced CPT objects.
        startpoint : str
            Start location name.
        endpoint : str
            End location name.
        band : float, default=1000.0
            Corridor width in meters.
        scale_factor : float, default=10.0
            Horizontal scale for CPT traces.
        extend_profile : bool, default=True
            Extend profile projection to start/end.
        show_annotations : bool, default=True
            Toggle annotations.
        general_layout : dict[Any, Any] or None, default=None
            Extra Plotly layout options.
        fillcolordict : dict[str, str] or None, default=None
            Soil color map.
        logwidth : float, default=100.0
            Width of profile traces.
        opacity : float, default=0.5
            Profile opacity.
        uniformcolor : str or None, default=None
            Single CPT trace color override.
        **kwargs
            Forwarded to groundhog plotting utilities.

        Returns
        -------
        dict[str, plotly.graph_objs.Figure]
            Dictionary containing the combined diagram.

        Examples
        --------
        >>> SoilPlot.plot_combined_fence([], [], "A", "B")  # doctest: +SKIP
        {'diagram': ...}
        """
        if fillcolordict is None:
            fillcolordict = {
                "SAND": "yellow",
                "CLAY": "brown",
                "SAND/CLAY": "orange",
            }
        if general_layout is None:
            general_layout = {}
        combined_fence_fig_1 = plot_combined_longitudinal_profile(
            cpts=cpts,
            profiles=profiles,
            latlon=True,
            start=startpoint,
            end=endpoint,
            band=band,
            scale_factor=scale_factor,
            logwidth=logwidth,
            opacity=opacity,
            extend_profile=extend_profile,
            show_annotations=show_annotations,
            uniformcolor=uniformcolor,
            fillcolordict=fillcolordict,
            general_layout=general_layout,
            **kwargs,
        )
        return {"diagram": combined_fence_fig_1}

    def plot_testlocations(self, return_fig: bool = False, **kwargs) -> Union[go.Figure, None]:
        """Plot test locations on an OpenStreetMap-backed scatter plot.

        Parameters
        ----------
        return_fig : bool, default=False
            If ``True``, return the figure instead of showing it.
        **kwargs
            Forwarded to :meth:`SoilAPI.get_testlocations`.

        Returns
        -------
        plotly.graph_objs.Figure or None
            Figure object when ``return_fig=True``, otherwise ``None``.

        Examples
        --------
        >>> from unittest.mock import Mock
        >>> import pandas as pd
        >>> api = Mock()
        >>> api.get_testlocations.return_value = {
        ...     "data": pd.DataFrame(
        ...         {
        ...             "northing": [50.0],
        ...             "easting": [2.0],
        ...             "title": ["T"],
        ...             "projectsite_name": ["P"],
        ...             "description": [""],
        ...         }
        ...     )
        ... }
        >>> fig = SoilPlot(api).plot_testlocations(return_fig=True)
        >>> fig.__class__.__name__
        'Figure'
        """
        testlocations = self.soil_api.get_testlocations(**kwargs)["data"]
        fig = px.scatter_mapbox(
            testlocations,
            lat="northing",
            lon="easting",
            hover_name="title",
            hover_data=["projectsite_name", "description"],
            zoom=10,
            height=500,
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        if return_fig:
            return fig
        else:
            fig.show()

    def plot_cpt_fence(
        self,
        cpt_df: pd.DataFrame,
        start: str,
        end: str,
        band: float = 1000.0,
        scale_factor: float = 10.0,
        extend_profile: bool = True,
        plotmap: bool = False,
        show_annotations: bool = True,
        general_layout: Union[dict[Any, Any], None] = None,
        uniformcolor: Union[str, None] = None,
        **kwargs,
    ) -> dict[str, Union[list[pd.DataFrame], go.Figure]]:
        """Create a fence diagram for selected CPTs.

        Parameters
        ----------
        cpt_df : pandas.DataFrame
            CPT summary rows.
        start : str
            Start location.
        end : str
            End location.
        band : float, default=1000.0
            Corridor width in meters.
        scale_factor : float, default=10.0
            Width scaling for CPT traces.
        extend_profile : bool, default=True
            Extend the profile line to start/end.
        plotmap : bool, default=False
            If ``True``, show map panel.
        show_annotations : bool, default=True
            Toggle annotations in the figure.
        general_layout : dict[Any, Any] or None, default=None
            Extra layout options.
        uniformcolor : str or None, default=None
            Single color for all CPT traces.
        **kwargs
            Forwarded to plotting utility.

        Returns
        -------
        dict[str, list[pandas.DataFrame] | plotly.graph_objs.Figure]
            Dictionary with loaded CPT objects and fence figure.

        Examples
        --------
        >>> import pandas as pd
        >>> from unittest.mock import Mock
        >>> plotter = SoilPlot(Mock())
        >>> sample = pd.DataFrame(
        ...     columns=[
        ...         "projectsite_name",
        ...         "location_name",
        ...         "title",
        ...         "test_type_name",
        ...         "easting",
        ...         "northing",
        ...         "elevation",
        ...     ]
        ... )
        >>> plotter.plot_cpt_fence(sample, "A", "B")  # doctest: +SKIP
        """
        if general_layout is None:
            general_layout = {}
        selected_cpts = cpt_df
        cpts = SoilDataProcessor.objects_to_list(selected_cpts, self.soil_api.get_cpttest_detail, "cpt")
        cpt_fence_fig_1 = plot_longitudinal_profile(
            cpts=cpts,
            latlon=True,
            start=start,
            end=end,
            band=band,
            scale_factor=scale_factor,
            extend_profile=extend_profile,
            plotmap=plotmap,
            show_annotations=show_annotations,
            general_layout=general_layout,
            uniformcolor=uniformcolor,
            **kwargs,
        )
        return {"cpts": cpts, "diagram": cpt_fence_fig_1}
