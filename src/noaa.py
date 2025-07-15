import time
import requests
import pandas as pd
from pathlib import Path
from typing import Generator, Dict, Any, Optional, List, Tuple
import heapq


BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"


def paginate(
        endpoint: str,
        params: Dict[str, Any],
        headers: Dict[str, str] = None,
        base_url: str = BASE_URL,
        limit: int = 1000,
        max_retries: int = 5,
        retry_backoff: int = 2
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield all results from a NOAA CDO endpoint by paging through results.
    Uses exponential backoff retry logic on HTTP errors.

    Args:
        endpoint: API endpoint (e.g., 'stations' or 'data').
        params: Query parameters for the request.
        headers: HTTP headers including API token.
        base_url: Base URL for the API. Defaults to BASE_URL.
        limit: Number of records per page. Defaults to 1000.
        max_retries: Maximum retry attempts. Defaults to 5.
        retry_backoff: Base backoff in seconds. Defaults to 2.

    Yields:
        Dict[str, Any]: Individual record from the API response.

    Raises:
        Exception: If all retry attempts fail.
    """
    offset = 1
    while True:
        print(f"Fetching {endpoint}, offset={offset}")
        p = {**params, "limit": limit, "offset": offset}
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(f"{base_url}/{endpoint}", params=p, headers=headers)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                print(f"  Retrieved {len(results)} records")
                break
            except Exception as e:
                if attempt == max_retries:
                    raise
                sleep_time = retry_backoff * 2 ** (attempt - 1)
                print(
                    f"  Warning: {endpoint} request failed (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
        if not results:
            break
        for item in results:
            yield item
        offset += limit


def get_all_stations(
        fips_code: str,
        dsid: str,
        dtys: list,
        syear: int,
        eyear: int,
        headers: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve metadata for all stations in a given FIPS region that report specified datatypes.

    Args:
        fips_code: FIPS code for the location (e.g., 'FIPS:19').
        dsid: Dataset ID (e.g., 'GHCND').
        dtys: List of datatype IDs (e.g., ['TMIN', 'TMAX']).
        syear: Start year for data coverage filter.
        eyear: End year for data coverage filter.
        headers: HTTP headers including API token.

    Returns:
        Mapping of station ID to station metadata.
    """
    params = {
        "datasetid": dsid,
        "datatypeid": dtys,
        "locationid": fips_code,
        "startdate": f"{syear}-01-01",
        "enddate": f"{eyear}-12-31"
    }
    stations = {s["id"]: s for s in paginate("stations", params, headers)}
    print(f"  Found {len(stations)} stations")
    return stations


def get_station_history(
        station_id: str,
        headers: Dict[str, str],
        base_url: str = BASE_URL,
        max_retries: int = 5,
        retry_backoff: int = 2
) -> Dict[str, Any]:
    """
    Retrieve operational date range and coverage for a specific station.

    Args:
        station_id: Unique station identifier (e.g., 'GHCND:USC00123456').
        headers: HTTP headers including API token.
        base_url: Base URL for the API. Defaults to BASE_URL.
        max_retries: Maximum retry attempts. Defaults to 5.
        retry_backoff: Base backoff in seconds. Defaults to 2.

    Returns:
        Dictionary containing 'mindate', 'maxdate', and 'coverage'.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(f"{base_url}/stations/{station_id}", headers=headers)
            resp.raise_for_status()
            meta = resp.json()
            print(
                f"  Station {station_id} active {meta.get('mindate')} to {meta.get('maxdate')} - Coverage: {meta.get('datacoverage')}")
            return {
                "mindate": meta.get("mindate"),
                "maxdate": meta.get("maxdate"),
                "coverage": float(meta.get("datacoverage", 0))
            }
        except Exception as e:
            if attempt == max_retries:
                raise
            sleep_time = retry_backoff * 2 ** (attempt - 1)
            print(
                f"  Warning: station history fetch failed (attempt {attempt}/{max_retries}): {e}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)


def preselect_best_stations(
        stations: Dict[str, Dict[str, Any]],
        syear: int,
        eyear: int,
        headers: Dict[str, str],
        max_stations: int = 20
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Select the best stations based on data coverage and availability for the required years.

    Args:
        stations: Dictionary of stations metadata.
        syear: Start year for the required data.
        eyear: End year for the required data.
        headers: HTTP headers including API token.
        max_stations: Maximum number of stations to select.

    Returns:
        List of tuple pairs (station_id, history) sorted by coverage and time span.
    """
    print(f"Preselecting best {max_stations} stations from {len(stations)} total stations...")

    # Track which years need coverage
    years_needed = set(range(syear, eyear + 1))

    # Create a priority queue to store stations by score
    station_scores = []

    # First pass - fetch history and calculate scores for all stations
    for i, (sid, metadata) in enumerate(stations.items()):
        print(f"[{i + 1}/{len(stations)}] Getting history for {sid}")

        try:
            history = get_station_history(sid, headers)
            start_y = int(history["mindate"][:4])
            end_y = int(history["maxdate"][:4])

            # Calculate years covered
            start_year = max(start_y, syear)
            end_year = min(end_y, eyear)
            years_covered = 0

            if end_year >= start_year:
                years_covered = end_year - start_year + 1

            # Score is a combination of coverage percentage and years spanned
            coverage_score = history["coverage"]
            years_score = years_covered / (eyear - syear + 1)
            score = (coverage_score * 0.7) + (years_score * 0.3)

            # Store as a tuple (-score for max heap, sid, history)
            heapq.heappush(station_scores, (-score, sid, history))

            # Rate limit API calls
            time.sleep(0.5)

            # Early exit optimization if we processed enough stations with high coverage
            if i >= max_stations * 2 and len(station_scores) >= max_stations:
                break

        except Exception as e:
            print(f"  Error getting history for {sid}: {e}")

    # Take the top N stations
    best_stations = []
    for _ in range(min(max_stations, len(station_scores))):
        if station_scores:
            _, sid, history = heapq.heappop(station_scores)
            best_stations.append((sid, history))

    print(f"Selected {len(best_stations)} best stations")
    return best_stations


def build_year_station_map(
        best_stations: List[Tuple[str, Dict[str, Any]]],
        syear: int,
        eyear: int
) -> Dict[int, str]:
    """
    Map each year to the best station for that year.

    Args:
        best_stations: List of (station_id, history) tuples.
        syear: Start year for the required data.
        eyear: End year for the required data.

    Returns:
        Dictionary mapping years to station IDs.
    """
    year_station_map = {}

    # Sort stations by coverage quality
    sorted_stations = sorted(best_stations, key=lambda x: x[1]["coverage"], reverse=True)

    # For each year, find the best station
    for year in range(syear, eyear + 1):
        for station_id, history in sorted_stations:
            start_y = int(history["mindate"][:4])
            end_y = int(history["maxdate"][:4])

            if start_y <= year <= end_y:
                year_station_map[year] = station_id
                break

    # Report coverage
    covered_years = len(year_station_map)
    total_years = eyear - syear + 1
    print(
        f"Year-station mapping complete: {covered_years}/{total_years} years covered ({covered_years / total_years * 100:.1f}%)")

    return year_station_map


def fetch_year_data(
        dsid: str,
        dtys: list,
        station_id: str,
        year: int,
        headers: Dict[str, str]
) -> pd.DataFrame:
    """
    Fetch daily observations for a given station and year.

    Args:
        dsid: Dataset ID (e.g., 'GHCND').
        dtys: List of datatype IDs to retrieve.
        station_id: Station identifier.
        year: Year for which to fetch data.
        headers: HTTP headers including API token.

    Returns:
        pd.DataFrame: DataFrame containing daily records.
    """
    params = {
        "datasetid": dsid,
        "datatypeid": dtys,
        "stationid": station_id,
        "startdate": f"{year}-01-01",
        "enddate": f"{year}-12-31"
    }
    records = list(paginate("data", params, headers))
    print(f"  Year {year}: downloaded {len(records)} records from station {station_id}")
    return pd.DataFrame(records)


def noaa_batch_data_extraction(
        state_fips: Dict[str, str],
        syear: int,
        eyear: int,
        dsid: str,
        dtys: list,
        data_folder: Path,
        headers: Dict[str, str],
        max_stations_per_state: int = 20
) -> None:
    """
    Fetch and save daily data for multiple states, using optimal station selection.

    Args:
        state_fips: Mapping of state names to FIPS codes.
        syear: Start year for extraction.
        eyear: End year for extraction.
        dsid: Dataset ID.
        dtys: List of datatype IDs.
        data_folder: Directory to save CSV files.
        headers: HTTP headers including API token.
        max_stations_per_state: Maximum number of stations to use per state.
    """
    data_folder.mkdir(parents=True, exist_ok=True)

    # Also create a metadata folder for saving station information
    metadata_folder = data_folder / "station_metadata"
    metadata_folder.mkdir(parents=True, exist_ok=True)

    for state, fips in state_fips.items():
        print(f"\nProcessing {state} ({fips})")

        # Get all potential stations
        stations = get_all_stations(fips, dsid, dtys, syear, eyear, headers)
        if not stations:
            print("  No stations available, skipping.")
            continue

        # Select the best stations
        best_stations = preselect_best_stations(stations, syear, eyear, headers, max_stations_per_state)

        # Build a year-to-station mapping
        year_station_map = build_year_station_map(best_stations, syear, eyear)

        # Save the station mapping for reference
        station_map_df = pd.DataFrame([
            {"year": year, "station_id": station_id}
            for year, station_id in year_station_map.items()
        ])
        station_map_df.to_csv(metadata_folder / f"{state.replace(' ', '_')}_station_map.csv", index=False)

        # Fetch data for each year from the assigned station
        df_years = []

        for year in range(syear, eyear + 1):
            if year not in year_station_map:
                print(f"  Year {year}: no station covers this year, skipping.")
                continue

            station_id = year_station_map[year]

            try:
                # Fetch data for this specific year
                df_year = fetch_year_data(dsid, dtys, station_id, year, headers)
                if len(df_year) > 0:
                    df_years.append(df_year)

            except Exception as e:
                print(f"  Error fetching year {year} from station {station_id}: {e}")

            # Rate limit API calls
            time.sleep(1)

        if df_years:
            df_all = pd.concat(df_years, ignore_index=True)
            out_file = data_folder / f"{state.replace(' ', '_')}.csv"
            df_all.to_csv(out_file, index=False)
            print(f"  Saved {len(df_all)} records to {out_file}")

            # Also save per-station metadata
            station_meta_df = pd.DataFrame([
                {
                    "station_id": sid,
                    "name": stations.get(sid, {}).get("name", "Unknown"),
                    "latitude": stations.get(sid, {}).get("latitude"),
                    "longitude": stations.get(sid, {}).get("longitude"),
                    "elevation": stations.get(sid, {}).get("elevation"),
                    "min_date": history.get("mindate"),
                    "max_date": history.get("maxdate"),
                    "coverage": history.get("coverage")
                }
                for sid, history in best_stations
            ])
            station_meta_df.to_csv(metadata_folder / f"{state.replace(' ', '_')}_stations.csv", index=False)
        else:
            print("  No data retrieved.")