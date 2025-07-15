import pandas as pd # type: ignore
from pathlib import Path
from functools import reduce


def load_price(commodity: str, tickers_dict: dict, data_folder: Path, subfolder: str='price') -> pd.DataFrame:
    """
    Load price data for a given commodity.

    Args:
        commodity: Commodity name.
        tickers_dict: Dictionary mapping commodity names to tickers.
        data_folder: Path to the data folder.
        subfolder: Subfolder name where the data is stored.

    Returns:
        DataFrame containing the date and closing price.

    """
    df= (
        pd.read_csv(
            data_folder / subfolder / f'{subfolder}_{tickers_dict[commodity]}.csv',
            sep=';'
        )[['Date', 'Close']]
        .rename(columns=str.lower) # type: ignore
        .rename(columns={
            'close': "_".join(commodity.split()).lower()
        })
    )
    return df


def load_weather(state: str, data_folder: Path, subfolder: str='nasa') -> pd.DataFrame:
    """
    Load weather data for a given commodity.

    Args:
        state: State name.
        data_folder: Path to the data folder.
        subfolder: Subfolder name where the data is stored.

    Returns:
        DataFrame containing the date and weather data.

    """
    if subfolder == 'nasa':
        df= (
            pd.read_csv(data_folder / subfolder / f'{state}_nasa_power.csv')
            .rename(columns=str.lower)
            .rename(columns={
                't2m_min': f'{state.lower()}_tmin',
                't2m_max': f'{state.lower()}_tmax',
                'prectotcorr': f'{state.lower()}_prc',
            })
        )
    elif subfolder == 'noaa':
        raise NotImplementedError("NOAA data loading is not implemented yet.")
    else:
        raise ValueError(f"Unknown subfolder/provider: {subfolder}")
    return df


def data_prep(
        comoddities: list,
        states: list,
        tickers_dict: dict,
        data_folder: Path,
        data_provider: str
) -> pd.DataFrame:
    """"""
    # Prepare price data.
    price_data = reduce(
        lambda x, y: x.merge(y, how='outer', on='date'),
        list(map(lambda c: load_price(c, tickers_dict, data_folder), comoddities))
    )

    # Prepare weather data.
    weather_data = reduce(
        lambda x, y: x.merge(y, how='outer', on='date'),
        list(map(lambda s: load_weather(s, data_folder, data_provider), states))
    )
    # Merge price and weather data.
    data = price_data.merge(weather_data, how='left', on='date')

    return data