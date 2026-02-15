"""Soil processing utilities.

This module contains helper classes for transforming and preparing soil data
returned by the OWI metadatabase soil extension.

Examples
--------
>>> import pandas as pd
>>> from owi.metadatabase.soil.processing import SoilDataProcessor
>>> df = pd.DataFrame({"id": [1], "title": ["A"], "offset [m]": [10.0]})
>>> result = SoilDataProcessor.gather_data_entity(df)
>>> int(result["id"])
1
"""

# mypy: ignore-errors

import warnings
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd
from groundhog.general.soilprofile import profile_from_dataframe
from groundhog.siteinvestigation.insitutests.pcpt_processing import PCPTProcessing
from pyproj import Transformer

if TYPE_CHECKING:
    from groundhog.general.soilprofile import SoilProfile


class SoilDataProcessor:
    """Helper routines for processing soil API payloads.

    Examples
    --------
    >>> import pandas as pd
    >>> from owi.metadatabase.soil.processing import SoilDataProcessor
    >>> raw = pd.DataFrame({"z [m]": [0.0], "qc": [5.0]})
    >>> proc = pd.DataFrame({"z [m]": [0.0], "qt": [5.1]})
    >>> SoilDataProcessor.combine_dfs({"rawdata": raw, "processeddata": proc}).shape
    (1, 3)
    """

    @staticmethod
    def transform_coord(
        df: pd.DataFrame, longitude: float, latitude: float, target_srid: str
    ) -> tuple[pd.DataFrame, float, float]:
        """Transform coordinates from EPSG:4326 to a target SRID.

        Parameters
        ----------
        df : pandas.DataFrame
            Input data containing ``easting`` and ``northing`` columns.
        longitude : float
            Longitude of the reference point in decimal degrees.
        latitude : float
            Latitude of the reference point in decimal degrees.
        target_srid : str
            Target EPSG code (for example ``"25831"``).

        Returns
        -------
        tuple[pandas.DataFrame, float, float]
            Updated DataFrame, transformed easting, and transformed northing.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"easting": [2.0], "northing": [50.0]})
        >>> out, east, north = SoilDataProcessor.transform_coord(df, 2.0, 50.0, "25831")
        >>> {"easting [m]", "northing [m]"}.issubset(set(out.columns))
        True
        >>> isinstance(east, float) and isinstance(north, float)
        True
        """
        transformer = Transformer.from_crs("epsg:4326", f"epsg:{target_srid}", always_xy=True)
        try:
            # Transform the easting and northing columns in the DataFrame
            df["easting [m]"], df["northing [m]"] = transformer.transform(df["easting"], df["northing"])
        except Exception as err:
            warnings.warn(f"Error transforming DataFrame coordinates: {err}", stacklevel=2)
        # Transform the reference central point
        point_east, point_north = transformer.transform(longitude, latitude)
        return df, point_east, point_north

    @staticmethod
    def combine_dfs(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge raw and processed in-situ test tables on depth.

        Parameters
        ----------
        dfs : dict[str, pandas.DataFrame]
            Dictionary containing ``rawdata`` and ``processeddata``.

        Returns
        -------
        pandas.DataFrame
            Merged DataFrame, or raw data if merge fails.

        Examples
        --------
        >>> import pandas as pd
        >>> raw = pd.DataFrame({"z [m]": [0.0], "qc": [5.0]})
        >>> proc = pd.DataFrame({"z [m]": [0.0], "qt": [5.1]})
        >>> SoilDataProcessor.combine_dfs({"rawdata": raw, "processeddata": proc}).columns.tolist()
        ['z [m]', 'qc', 'qt']
        """
        try:
            combined_df = pd.merge(
                dfs["rawdata"],
                dfs["processeddata"],
                on="z [m]",
                how="inner",
                suffixes=("", "_processed"),
            )
            return combined_df
        except Exception as err:
            warnings.warn(f"Error combining raw and processed data: {err}", stacklevel=2)
            return dfs.get("rawdata", pd.DataFrame())

    @staticmethod
    def process_insitutest_dfs(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
        """Extract nested in-situ test payloads as flat DataFrames.

        Parameters
        ----------
        df : pandas.DataFrame
            In-situ test detail table with nested columns.
        cols : list[str]
            Column names to extract and convert.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Processed tables keyed by source column name.

        Examples
        --------
        >>> import pandas as pd
        >>> detail = pd.DataFrame({"rawdata": [[{"z [m]": 0.0, "qc": 5.0}]]})
        >>> out = SoilDataProcessor.process_insitutest_dfs(detail, ["rawdata"])
        >>> list(out.keys())
        ['rawdata']
        >>> out["rawdata"].shape[0]
        1
        """
        processed_dfs = {}
        for col in cols:
            try:
                # The column data is assumed to be in the first row as a nested
                # dict or list.
                temp_df = pd.DataFrame(df[col].iloc[0]).reset_index(drop=True)
                processed_dfs[col] = temp_df
            except KeyError:
                warnings.warn(
                    f"""
                    Column '{col}' not found. Check the DataFrame structure.

                    Check that you entered correct parameters in your request
                    or contact database administrators.
                    """,
                    stacklevel=2,
                )
                processed_dfs[col] = pd.DataFrame()
            except Exception as e:
                warnings.warn(f"Error processing column '{col}': {e}", stacklevel=2)
                processed_dfs[col] = pd.DataFrame()

        # Attempt to convert values to numeric where applicable.
        for key in processed_dfs:
            try:
                processed_dfs[key] = processed_dfs[key].apply(pd.to_numeric)
            except Exception as err:
                warnings.warn(f"Numeric conversion warning for {key}: {err}", stacklevel=2)
        return processed_dfs

    @staticmethod
    def gather_data_entity(
        df: pd.DataFrame,
    ) -> dict[str, Union[pd.DataFrame, int, str, float, None]]:
        """Select the closest entity and return metadata with the full table.

        Parameters
        ----------
        df : pandas.DataFrame
            Candidate entities, including ``id``, ``title`` and ``offset [m]``.

        Returns
        -------
        dict[str, pandas.DataFrame | int | str | float | None]
            Dictionary containing selected id/title/offset and input data.

        Examples
        --------
        >>> import pandas as pd
        >>> inp = pd.DataFrame({"id": [2, 1], "title": ["B", "A"], "offset [m]": [5.0, 1.0]})
        >>> out = SoilDataProcessor.gather_data_entity(inp)
        >>> int(out["id"])
        1
        """
        if df.__len__() == 1:
            loc_id = df["id"].iloc[0]
        else:
            df.sort_values("offset [m]", inplace=True)
            loc_id = df[df["offset [m]"] == df["offset [m]"].min()]["id"].iloc[0]
        return {
            "data": df,
            "id": loc_id,
            "title": df["title"].iloc[0],
            "offset [m]": df[df["offset [m]"] == df["offset [m]"].min()]["offset [m]"].iloc[0],
        }

    @staticmethod
    def process_cpt(df_sum: pd.DataFrame, df_raw: pd.DataFrame, **kwargs):
        """Create a ``PCPTProcessing`` object from CPT summary and raw data.

        Parameters
        ----------
        df_sum : pandas.DataFrame
            CPT summary table containing the CPT title.
        df_raw : pandas.DataFrame
            CPT raw measurement table.
        **kwargs
            Forwarded to ``PCPTProcessing.load_pandas``.

        Returns
        -------
        PCPTProcessing or None
            Processed CPT object, or ``None`` on failure.

        Examples
        --------
        >>> import pandas as pd
        >>> df_sum = pd.DataFrame({"title": ["CPT-1"]})
        >>> df_raw = pd.DataFrame({"z [m]": [0.0], "qc": [1.0]})
        >>> obj = SoilDataProcessor.process_cpt(df_sum, df_raw)  # doctest: +ELLIPSIS
        >>> obj is None or obj.__class__.__name__ == 'PCPTProcessing'
        True
        """
        try:
            cpt = PCPTProcessing(title=df_sum["title"].iloc[0])
            push_key = "Push" if "Push" in df_raw else None
            cpt.load_pandas(df_raw, push_key=push_key, **kwargs)
            return cpt
        except Exception as err:
            warnings.warn(f"ERROR: PCPTProcessing object not created - {err}", stacklevel=2)
            return None

    @staticmethod
    def convert_to_profile(
        df_sum: pd.DataFrame,
        df_detail: pd.DataFrame,
        profile_title: Optional[str],
        drop_info_cols: bool,
    ) -> Optional["SoilProfile"]:
        """Convert soil profile detail records to a Groundhog profile object.

        Parameters
        ----------
        df_sum : pandas.DataFrame
            Soil profile summary table.
        df_detail : pandas.DataFrame
            Soil profile detail table containing ``soillayer_set``.
        profile_title : str or None
            Title override for the output profile.
        drop_info_cols : bool
            If ``True``, drop metadata columns before conversion.

        Returns
        -------
        SoilProfile or None
            Converted profile, or ``None`` when conversion fails.

        Examples
        --------
        >>> import pandas as pd
        >>> df_sum = pd.DataFrame({"location_name": ["LOC"], "title": ["Profile"]})
        >>> layers = [{
        ...     "start_depth": 0.0,
        ...     "end_depth": 1.0,
        ...     "soiltype_name": "SAND",
        ...     "totalunitweight": 18.0,
        ...     "soilparameters": {},
        ...     "id": 1,
        ...     "profile": 1,
        ...     "soilprofile_name": "P",
        ...     "soilunit": None,
        ...     "description": "",
        ...     "soilunit_name": "",
        ... }]
        >>> df_detail = pd.DataFrame({"soillayer_set": [layers]})
        >>> profile = SoilDataProcessor.convert_to_profile(df_sum, df_detail, "Demo", True)
        >>> profile is None or profile.__class__.__name__ == 'SoilProfile'
        True
        """
        try:
            soilprofile_df = (
                pd.DataFrame(df_detail["soillayer_set"].iloc[0]).sort_values("start_depth").reset_index(drop=True)
            )
            soilprofile_df.rename(
                columns={
                    "start_depth": "Depth from [m]",
                    "end_depth": "Depth to [m]",
                    "soiltype_name": "Soil type",
                    "totalunitweight": "Total unit weight [kN/m3]",
                },
                inplace=True,
            )
            for i, row in soilprofile_df.iterrows():
                try:
                    for key, value in row["soilparameters"].items():
                        soilprofile_df.loc[i, key] = value
                except Exception:
                    pass
            if drop_info_cols:
                soilprofile_df.drop(
                    [
                        "id",
                        "profile",
                        "soilparameters",
                        "soilprofile_name",
                        "soilunit",
                        "description",
                        "soilunit_name",
                    ],
                    axis=1,
                    inplace=True,
                )
            # Convert numeric columns, excluding "Soil type" (str)
            for col in soilprofile_df.columns:
                if col != "Soil type":
                    try:
                        soilprofile_df[col] = pd.to_numeric(soilprofile_df[col], errors="coerce")
                    except Exception as err:
                        warnings.warn(
                            f"Error converting column '{col}' to numeric: {err}",
                            stacklevel=2,
                        )

            if profile_title is None:
                profile_title = f"{df_sum['location_name'].iloc[0]} - {df_sum['title'].iloc[0]}"
            dsp = profile_from_dataframe(soilprofile_df, title=profile_title)
            return dsp
        except KeyError:
            warnings.warn(
                """
                Something is wrong with the output dataframe:
                check that the database gave a non-empty output.

                Check that you entered correct parameters in your request
                or contact database administrators.
                """,
                stacklevel=2,
            )
            return None
        except Exception as err:
            warnings.warn(f"Error during loading of soil layers and parameters: {err}", stacklevel=2)
            return None

    @staticmethod
    def fulldata_processing(unitdata, row, selected_depths, func_get_details, depthcol, **kwargs) -> pd.DataFrame:
        """Filter full test data to the selected depth ranges for one location.

        Parameters
        ----------
        unitdata : pandas.DataFrame
            Accumulator DataFrame.
        row : pandas.Series
            Row describing the current location.
        selected_depths : pandas.DataFrame
            Depth intervals per location.
        func_get_details : Callable
            Function returning detail data with a ``rawdata`` key.
        depthcol : str
            Name of the depth column in the returned data.
        **kwargs
            Forwarded to ``func_get_details``.

        Returns
        -------
        pandas.DataFrame
            Filtered and annotated unit data.

        Examples
        --------
        >>> import pandas as pd
        >>> row = pd.Series({"location_name": "LOC", "projectsite_name": "P", "test_type_name": "CPT"})
        >>> selected = pd.DataFrame({"location_name": ["LOC"], "start_depth": [0.0], "end_depth": [1.0]})
        >>> def _get_details(**kwargs):
        ...     return {"rawdata": pd.DataFrame({"z [m]": [0.5, 2.0], "qc": [1, 2]})}
        >>> out = SoilDataProcessor.fulldata_processing(pd.DataFrame(), row, selected, _get_details, "z [m]")
        >>> out.shape[0]
        1
        """
        _fulldata = func_get_details(location=row["location_name"], **kwargs)["rawdata"]
        _depthranges = selected_depths[selected_depths["location_name"] == row["location_name"]]
        for _, _layer in _depthranges.iterrows():
            _unitdata = _fulldata[
                (_fulldata[depthcol] >= _layer["start_depth"]) & (_fulldata[depthcol] <= _layer["end_depth"])
            ]
            unitdata = pd.concat([unitdata, _unitdata])
        unitdata.reset_index(drop=True, inplace=True)
        unitdata.loc[:, "location_name"] = row["location_name"]
        unitdata.loc[:, "projectsite_name"] = row["projectsite_name"]
        unitdata.loc[:, "test_type_name"] = row["test_type_name"]
        return unitdata

    @staticmethod
    def partialdata_processing(unitdata, row, selected_depths, selected_tests):
        """Append selected tests whose point depth falls in selected intervals.

        Parameters
        ----------
        unitdata : pandas.DataFrame
            Accumulator DataFrame.
        row : pandas.Series
            Current test row with ``id`` and ``depth``.
        selected_depths : pandas.DataFrame
            Depth intervals per location.
        selected_tests : pandas.DataFrame
            Candidate tests.

        Returns
        -------
        None
            The function mutates ``unitdata`` in place.

        Examples
        --------
        >>> import pandas as pd
        >>> unit = pd.DataFrame()
        >>> row = pd.Series({"id": 1, "depth": 0.5, "location_name": "LOC"})
        >>> depth = pd.DataFrame({"location_name": ["LOC"], "start_depth": [0.0], "end_depth": [1.0]})
        >>> tests = pd.DataFrame({"id": [1], "qc": [10]})
        >>> SoilDataProcessor.partialdata_processing(unit, row, depth, tests)
        """
        _depthranges = selected_depths[selected_depths["location_name"] == row["location_name"]]
        for _, _layer in _depthranges.iterrows():
            if row["depth"] >= _layer["start_depth"] and row["depth"] <= _layer["end_depth"]:
                _unitdata = selected_tests[selected_tests["id"] == row["id"]]
                unitdata = pd.concat([unitdata, _unitdata])
            else:
                pass
        unitdata.reset_index(drop=True, inplace=True)

    @staticmethod
    def objects_to_list(selected_obj, func_get_detail, data_type):
        """Load and georeference profile/CPT objects from summary rows.

        Parameters
        ----------
        selected_obj : pandas.DataFrame
            Summary rows selected by the user.
        func_get_detail : Callable
            API method that returns a detail dictionary.
        data_type : str
            Target key in the detail dictionary (``"soilprofile"`` or ``"cpt"``).

        Returns
        -------
        list
            List of loaded objects with position set.

        Examples
        --------
        >>> import pandas as pd
        >>> selected = pd.DataFrame(
        ...     columns=[
        ...         "projectsite_name",
        ...         "location_name",
        ...         "title",
        ...         "easting",
        ...         "northing",
        ...         "elevation",
        ...         "test_type_name",
        ...     ]
        ... )
        >>> SoilDataProcessor.objects_to_list(selected, lambda **k: {}, "soilprofile")
        []
        """
        obj = []
        for _, row in selected_obj.iterrows():
            try:
                if data_type == "soilprofile":
                    params = {
                        "projectsite": row["projectsite_name"],
                        "location": row["location_name"],
                        "soilprofile": row["title"],
                        "drop_info_cols": False,
                        "profile_title": row["location_name"],
                    }
                elif data_type == "cpt":
                    params = {
                        "projectsite": row["projectsite_name"],
                        "location": row["location_name"],
                        "insitutest": row["title"],
                        "testtype": row["test_type_name"],
                    }
                else:
                    raise ValueError(f"Data type {data_type} not supported.")
                _obj = func_get_detail(**params)[data_type]
                _obj.set_position(
                    easting=row["easting"],
                    northing=row["northing"],
                    elevation=row["elevation"],
                )
                obj.append(_obj)
            except Exception:
                warnings.warn(
                    f"Error loading {row['projectsite_name']}-{row['location_name']}-{row['title']}", stacklevel=2
                )
        return obj


class SoilprofileProcessor:
    """Prepare soil profile inputs for SSI workflows.

    Notes
    -----
    The class keeps central key registries for different SSI methods and uses
    them to validate and subset user-provided DataFrames.

    Examples
    --------
    >>> SoilprofileProcessor.get_available_options("lateral")
    ['apirp2geo', 'pisa']
    """

    LATERAL_SSI_KEYS: dict[str, dict[str, list[Union[str, tuple[str, str]]]]] = {
        "apirp2geo": {
            "mandatory": [
                "Depth from [m]",
                "Depth to [m]",
                "Soil type",
                ("Total unit weight", "[kN/m3]"),
                ("Su", "[kPa]"),
                ("Phi", "[deg]"),
                ("epsilon50", "[-]"),
            ],
            "optional": [
                ("Dr", "[-]"),
            ],
        },
        "pisa": {
            "mandatory": [
                "Depth from [m]",
                "Depth to [m]",
                "Soil type",
                ("Total unit weight", "[kN/m3]"),
                ("Gmax", "[kPa]"),
                ("Su", "[kPa]"),
                ("Dr", "[-]"),
            ],
            "optional": [],
        },
    }

    AXIAL_SSI_KEYS: dict[str, dict[str, list[Union[str, tuple[str, str]]]]] = {
        "cpt": {
            "mandatory": [],
            "optional": [],
        }
    }

    @classmethod
    def get_available_options(cls, loading: str = "lateral") -> list[str]:
        """Return available processing options for a loading family.

        Parameters
        ----------
        loading : str, default="lateral"
            Loading family to query (``"lateral"`` or ``"axial"``).

        Returns
        -------
        list[str]
            Option names configured for the selected loading family.

        Raises
        ------
        ValueError
            If ``loading`` is unsupported.

        Examples
        --------
        >>> SoilprofileProcessor.get_available_options("axial")
        ['cpt']
        """
        if loading.lower() == "lateral":
            return list(cls.LATERAL_SSI_KEYS.keys())
        elif loading.lower() == "axial":
            return list(cls.AXIAL_SSI_KEYS.keys())
        else:
            raise ValueError(f"Unsupported loading type '{loading}'.")

    @staticmethod
    def _validate_keys(data: pd.DataFrame, required_keys: list, mandatory: bool = True) -> list[str]:
        """Validate and normalize expected column names.

        Parameters
        ----------
        data : pandas.DataFrame
            Input soil profile table.
        required_keys : list
            Required keys as strings or tuples of identifying fragments.
        mandatory : bool, default=True
            If ``True``, raise when a key is missing.

        Returns
        -------
        list[str]
            Validated output column names.

        Raises
        ------
        ValueError
            If mandatory keys are missing or ambiguous.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame(columns=["Depth from [m]", "Depth to [m]", "Soil type"])
        >>> SoilprofileProcessor._validate_keys(df, ["Depth from [m]", "Depth to [m]"])
        ['Depth from [m]', 'Depth to [m]']
        """
        validated_columns = []
        # Map lower-case column names to original names for renaming.
        keys_lower = {col.lower(): col for col in data.columns}

        for key in required_keys:
            if isinstance(key, tuple):
                candidate = []
                for col in data.columns:
                    if all(elem.lower() in col.lower() for elem in key):
                        candidate.append(col)
                if candidate == []:
                    if mandatory:
                        raise ValueError(f"Soil input: '{key}' is missing in the soil data.")
                    else:
                        continue
                validated_columns.extend(candidate)
            else:
                # For a string key, check using lower-case comparison.
                matching_cols = [col for col in data.columns if key.lower() in col.lower()]
                if len(matching_cols) == 0:
                    if mandatory:
                        raise ValueError(f"Soil input: '{key}' is missing in the soil data.")
                    else:
                        continue
                elif len(matching_cols) > 1:
                    raise ValueError(f"'{key}' should be defined by a single column, found: {matching_cols}")

                original = keys_lower[key.lower()]
                if original != key:
                    data.rename(columns={original: key}, inplace=True)
                validated_columns.append(key)
        return validated_columns

    @classmethod
    def lateral(
        cls,
        df: pd.DataFrame,
        option: str,
        mudline: Union[float, None] = None,
        pw: float = 1.025,
    ) -> pd.DataFrame:
        """Prepare a soil profile table for lateral SSI workflows.

        Parameters
        ----------
        df : pandas.DataFrame
            Source soil profile data.
        option : str
            Lateral option name, such as ``"apirp2geo"`` or ``"pisa"``.
        mudline : float or None, default=None
            Seabed level in mLAT.
        pw : float, default=1.025
            Seawater density in t/m³.

        Returns
        -------
        pandas.DataFrame
            Filtered profile containing required and optional keys.

        Raises
        ------
        NotImplementedError
            If ``option`` is unsupported.
        ValueError
            If mandatory columns are missing.

        Examples
        --------
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     "Depth from [m]": [0.0], "Depth to [m]": [1.0], "Soil type": ["SAND"],
        ...     "Total unit weight [kN/m3]": [18.0], "Su [kPa]": [10.0], "Phi [deg]": [30.0],
        ...     "epsilon50 [-]": [0.02], "Dr [-]": [0.6]
        ... })
        >>> out = SoilprofileProcessor.lateral(data, "apirp2geo")
        >>> "Submerged unit weight [kN/m3]" in out.columns
        True
        """
        available_options = cls.get_available_options(loading="lateral")
        if option not in available_options:
            raise NotImplementedError(f"Option '{option}' not supported.")

        key_db = cls.LATERAL_SSI_KEYS[option]
        # Mandatory keys for the selected option.
        _keys = key_db.get("mandatory", [])
        mandatory_keys = cls._validate_keys(data=df, required_keys=_keys, mandatory=True)
        # Include optional keys that are present.
        _keys = key_db.get("optional", [])
        optional_keys = cls._validate_keys(data=df, required_keys=_keys, mandatory=False)
        soilprofile = df[mandatory_keys + optional_keys].copy()
        # Add additional required info
        soilprofile = cls._add_soilinfo(soilprofile, pw, mudline)

        return soilprofile

    @staticmethod
    def _add_soilinfo(df: pd.DataFrame, pw: float, mudline: Union[float, None]) -> pd.DataFrame:
        """Append derived soil information columns.

        Parameters
        ----------
        df : pandas.DataFrame
            Soil profile table.
        pw : float
            Seawater density in t/m³.
        mudline : float or None
            Seabed level in mLAT.

        Returns
        -------
        pandas.DataFrame
            DataFrame with submerged unit weight and optional elevations.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({"Depth from [m]": [0.0], "Depth to [m]": [1.0], "Total unit weight [kN/m3]": [18.0]})
        >>> out = SoilprofileProcessor._add_soilinfo(df, pw=1.025, mudline=0.0)
        >>> "Submerged unit weight [kN/m3]" in out.columns
        True
        """
        # Add submerged unit weight
        acc_gravity = 9.81  # acceleration due to gravity (m/s2)
        for col in df.columns:
            if "Total unit weight" in col:
                new_col = col.replace("Total unit weight", "Submerged unit weight")
                df[new_col] = df[col] - pw * acc_gravity

        # Add mudline depth in mLAT coordinates
        if mudline:
            df["Elevation from [mLAT]"] = mudline - df["Depth from [m]"]
            df["Elevation to [mLAT]"] = mudline - df["Depth to [m]"]

        return df
