import pandas as pd # type: ignore
import numpy as np # type: ignore
import json


def check_missing_values(df):
    """Check for missing values and return series with >10% missing data."""
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    
    flagged_columns = missing_pct[missing_pct > 10].index.tolist()
    
    print("Missing values summary:")
    for col in df.columns:
        print(f"  {col}: {missing_pct[col]:.2f}% missing")
    
    if flagged_columns:
        print(f"\nColumns flagged for manual review (>10% missing):")
        for col in flagged_columns:
            print(f"  {col}: {missing_pct[col]:.2f}% missing")
    
    return flagged_columns

def fill_short_gaps(series, max_gap=5):
    """Fill gaps of up to max_gap days using forward/backward filling."""
    # Create a mask of NaN values
    mask = series.isna()
    
    # Find runs of NaN values
    mask_shift = mask.shift(1, fill_value=False)
    run_starts = np.where(mask & ~mask_shift)[0]
    
    if len(run_starts) == 0:
        return series
    
    run_lengths = []
    for start in run_starts:
        length = 0
        pos = start
        while pos < len(series) and mask.iloc[pos]:
            length += 1
            pos += 1
        run_lengths.append(length)
    
    # Fill short gaps using forward fill then backward fill
    filled_series = series.copy()
    for start, length in zip(run_starts, run_lengths):
        if length <= max_gap:
            # Get the indices for this run
            indices = range(start, start + length)
            # Try forward fill first, then backward fill for any remaining NaNs
            for idx in indices:
                if pd.isna(filled_series.iloc[idx]) and idx > 0:
                    filled_series.iloc[idx] = filled_series.iloc[idx - 1]
            
            # Backward fill any remaining NaNs
            for idx in reversed(indices):
                if pd.isna(filled_series.iloc[idx]) and idx < len(filled_series) - 1:
                    filled_series.iloc[idx] = filled_series.iloc[idx + 1]
    
    return filled_series

def interpolate_long_gaps(series, max_gap=5):
    """Interpolate gaps longer than max_gap days."""
    # Create a mask of NaN values
    mask = series.isna()
    
    # Find runs of NaN values
    mask_shift = mask.shift(1, fill_value=False)
    run_starts = np.where(mask & ~mask_shift)[0]
    
    if len(run_starts) == 0:
        return series
    
    # Interpolate long gaps
    filled_series = series.copy()
    for start in run_starts:
        length = 0
        pos = start
        while pos < len(series) and mask.iloc[pos]:
            length += 1
            pos += 1
        
        if length > max_gap:
            # Get the indices for this run
            indices = range(start, start + length)
            # Get values before and after the gap
            if start > 0 and start + length < len(series):
                before_val = filled_series.iloc[start - 1]
                after_val = filled_series.iloc[start + length]
                
                # Linear interpolation
                if not pd.isna(before_val) and not pd.isna(after_val):
                    for i, idx in enumerate(indices):
                        filled_series.iloc[idx] = before_val + (after_val - before_val) * (i + 1) / (length + 1)
    
    return filled_series

def winsorize_outliers(series, window=30, threshold=3):
    """Winsorize outliers based on rolling Z-score."""
    # Calculate rolling mean and std
    rolling_mean = series.rolling(window=window, center=True).mean()
    rolling_std = series.rolling(window=window, center=True).std()
    
    # Calculate Z-scores
    z_scores = (series - rolling_mean) / rolling_std
    
    # Identify outliers
    outliers = (z_scores.abs() > threshold) & ~z_scores.isna()
    
    # Winsorize outliers
    winsorized = series.copy()
    winsorized[outliers & (z_scores > 0)] = rolling_mean[outliers & (z_scores > 0)] + threshold * rolling_std[outliers & (z_scores > 0)]
    winsorized[outliers & (z_scores < 0)] = rolling_mean[outliers & (z_scores < 0)] - threshold * rolling_std[outliers & (z_scores < 0)]
    
    print(f"Winsorized {outliers.sum()} outliers in {series.name}")
    
    return winsorized

def preprocess_data(df, states, commodity='corn'):
    """Preprocess the entire dataset."""
    print("Starting data preprocessing...")
    
    # Check for missing values
    flagged_columns = check_missing_values(df)
    
    # Process each column
    cleaned_df = df.copy()
    for column in df.columns:
        print(f"Processing column: {column}")
        
        # Fill short gaps
        series = fill_short_gaps(df[column])
        
        # Interpolate longer gaps
        series = interpolate_long_gaps(series)
        
        # Winsorize outliers
        if not series.isna().any():  # Only if no NaNs remain
            series = winsorize_outliers(series)
        
        cleaned_df[column] = series
    
    # Calculate price spreads
    if all(col in cleaned_df.columns for col in [commodity, f'{commodity}_oil']):
        cleaned_df[f'{commodity}_spread'] = cleaned_df[f'{commodity}_oil'] - cleaned_df[commodity]
    
    # Calculate weather anomalies (difference from seasonal average)
    for state in states:
        # Temperature anomalies
        if f'{state}_tmin' in cleaned_df.columns and f'{state}_tmax' in cleaned_df.columns:
            # Average temperature
            cleaned_df[f'{state}_tavg'] = (cleaned_df[f'{state}_tmin'] + cleaned_df[f'{state}_tmax']) / 2
            
            # Calculate daily climatology (long-term average for each day of year)
            climatology = cleaned_df[f'{state}_tavg'].groupby(cleaned_df.index.dayofyear).mean()
            
            # Calculate anomalies
            doy = cleaned_df.index.dayofyear
            cleaned_df[f'{state}_temp_anomaly'] = cleaned_df[f'{state}_tavg'] - [climatology[d] for d in doy]
            
            # 30-day rolling temperature anomaly
            cleaned_df[f'{state}_temp_anomaly_30d'] = cleaned_df[f'{state}_temp_anomaly'].rolling(window=30).mean()
        
        # Precipitation anomalies
        if f'{state}_prc' in cleaned_df.columns:
            # Calculate daily climatology
            prc_climatology = cleaned_df[f'{state}_prc'].groupby(cleaned_df.index.dayofyear).mean()
            
            # Calculate anomalies
            cleaned_df[f'{state}_prc_anomaly'] = cleaned_df[f'{state}_prc'] - [prc_climatology[d] for d in doy] # type: ignore
            
            # 30-day rolling precipitation anomaly
            cleaned_df[f'{state}_prc_anomaly_30d'] = cleaned_df[f'{state}_prc_anomaly'].rolling(window=30).mean()
    
    # Save metadata
    metadata = {
        'date_range': {
            'start': cleaned_df.index.min().strftime('%Y-%m-%d'),
            'end': cleaned_df.index.max().strftime('%Y-%m-%d')
        },
        'data_quality': {
            'flagged_columns': flagged_columns,
            'missing_values_after_cleaning': cleaned_df.isna().sum().to_dict()
        },
        'columns': list(cleaned_df.columns)
    }
    
    print("Data preprocessing complete.")
    return cleaned_df, metadata

def save_cleaned_data(df, metadata, data_folder, commodity='corn'):
    """Save cleaned data to Parquet and metadata to JSON."""
    data_folder.mkdir(parents=True, exist_ok=True)
    # Save to Parquet
    df.to_parquet(data_folder/  f'{commodity}_weather_cleaned.parquet')
    
    # Save metadata to JSON
    with open(data_folder /  f'{commodity}_weather_cleaned_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Saved cleaned data to {(data_folder / f'{commodity}_weather_cleaned.parquet').as_posix()}")
    print(f"Saved metadata saved to {(data_folder /  f'{commodity}_weather_cleaned_metadata.json').as_posix()}")