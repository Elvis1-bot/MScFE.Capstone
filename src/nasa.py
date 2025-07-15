import time
import requests # type: ignore
import pandas as pd # type: ignore
from pathlib import Path
from typing import Dict, Any


def request_with_retry(
    url: str,
    params: Dict[str, Any],
    headers: Dict[str, str] = None, # type: ignore
    max_retries: int = 5,
    backoff_factor: int = 2
) -> Dict[str, Any]: # type: ignore
    """
    Perform an HTTP GET with retries and exponential backoff.

    Args:
        url: Full API endpoint URL.
        params: Query parameters.
        headers: HTTP headers. Defaults to None.
        max_retries: Maximum retry attempts. Defaults to 5.
        backoff_factor: Base backoff in seconds. Defaults to 2.

    Returns:
        Parsed JSON response.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if attempt == max_retries:
                raise
            time.sleep(backoff_factor * 2 ** (attempt - 1))


def fetch_power_data(
    lat: float,
    lon: float,
    base_url: str = "https://power.larc.nasa.gov/api/temporal/daily/point",
    default_params: Dict[str, Any] = None, # type: ignore
    headers: Dict[str, str] = None, # type: ignore
    max_retries: int = 5,
    backoff_factor: int = 2
) -> pd.DataFrame:
    """
    Fetch daily NASA POWER data for specific coordinates.

    Args:
        lat: Latitude.
        lon: Longitude.
        base_url: NASA POWER API endpoint. Defaults to the daily-point URL.
        default_params: Default query parameters. Defaults to None.
        headers: HTTP headers. Defaults to None.
        max_retries: Retry attempts for HTTP. Defaults to 5.
        backoff_factor: Base backoff in seconds. Defaults to 2.

    Returns:
        DataFrame of daily values with a 'date' column.
    """
    params = default_params.copy()
    params.update({"latitude":lat,"longitude":lon})
    data = request_with_retry(
        url=base_url,
        params=params,
        headers=headers,
        max_retries=max_retries,
        backoff_factor=backoff_factor
    )
    parameter_data = data.get("properties", {}).get("parameter", {})
    df = pd.DataFrame(parameter_data)
    df["date"] = pd.to_datetime(df.index, format="%Y%m%d")
    df.reset_index(drop=True, inplace=True)
    return df


def nasa_batch_data_extraction(
    state_coords: Dict[str, Dict[str, float]],
    data_folder: Path,
    base_url: str = "https://power.larc.nasa.gov/api/temporal/daily/point",
    default_params: Dict[str, Any] = None, # type: ignore
    headers: Dict[str, str] = None, # type: ignore
    max_retries: int = 5,
    backoff_factor: int = 2
) -> None:
    """
    Extract and save NASA POWER daily data for all states.

    Args:
        state_coords: Mapping of states to lat/lon. Defaults to state_coords.
        data_folder: Directory to save CSVs. Defaults to data_folder.
        base_url: NASA POWER endpoint. Defaults to daily-point URL.
        default_params: Default API params. Defaults to None.
        headers: HTTP headers. Defaults to None.
        max_retries: Retry attempts for HTTP. Defaults to 5.
        backoff_factor: Base backoff in seconds. Defaults to 2.
    """
    data_folder.mkdir(parents=True, exist_ok=True)

    for state, coord in state_coords.items():
        lat = coord["lat"]
        lon = coord["lon"]
        print(f"Processing {state} ({lat},{lon})")
        try:
            df = fetch_power_data(
                lat=lat,
                lon=lon,
                base_url=base_url,
                default_params=default_params,
                headers=headers,
                max_retries=max_retries,
                backoff_factor=backoff_factor
            )
            output_file = data_folder / f"{state.replace(' ','_')}_nasa_power.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved data for {state} to {output_file}")
        except Exception as exc:
            print(f"Failed to fetch data for {state}: {exc}")
