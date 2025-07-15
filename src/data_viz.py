import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from scipy import signal # type: ignore


def plot_time_series(
    df,
    data_folder,
    spread_col='corn_spread',
    anomaly_col='iowa_temp_anomaly_30d',
    title='Commodity Price Spread vs Weather Anomaly',
):
    """Plot time series of price spread and weather anomaly."""
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plotting the first series.
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price Spread', color=color)

    ax1.plot(df.index, df[spread_col], color=color, label=spread_col)
    ax1.tick_params(axis='y', labelcolor=color)

    # Creating a second.
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Weather Anomaly', color=color)

    ax2.plot(df.index, df[anomaly_col], color=color, label=anomaly_col)
    ax2.tick_params(axis='y', labelcolor=color)

    # Set the title on ax1 for better practice with subplots
    ax1.set_title(title)

    # Add legend.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    fig.tight_layout()

    # Save figure.
    file_name = f'time_series_{spread_col}_{anomaly_col}.png'
    data_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(data_folder / file_name, dpi=300, bbox_inches='tight')
    print(f"Saved time series plot to {(data_folder / file_name).as_posix()}")

    plt.show()


def plot_spectral_analysis(
    df,
    data_folder,
    column='corn_spread',
    title='Spectral Analysis'
):
    """Plot spectral analysis using Fast Fourier Transform."""
    # Remove NaN values.
    series = df[column].dropna()

    # Detrend series.
    detrended = signal.detrend(series.values)

    # Compute FFT.
    fft_result = np.fft.fft(detrended)
    fft_freq = np.fft.fftfreq(len(detrended), d=1)  # Assuming daily data

    # Consider only positive frequencies and compute amplitude.
    positive_freq_idx = fft_freq > 0
    frequencies = fft_freq[positive_freq_idx]
    amplitudes = np.abs(fft_result[positive_freq_idx])

    # Convert to periods (in days).
    periods = 1 / frequencies

    # Plot.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(periods, amplitudes)
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)

    # Highlight key periods (annual, semi-annual).
    annual = 365
    semi_annual = 182.5

    # Find nearest points to these cycles.
    annual_idx = np.argmin(np.abs(periods - annual))
    semi_annual_idx = np.argmin(np.abs(periods - semi_annual))

    # Highlight these points.
    ax.axvline(x=periods[annual_idx], color='r', linestyle='--', alpha=0.7)
    ax.axvline(x=periods[semi_annual_idx], color='g', linestyle='--', alpha=0.7)

    # Add annotations.
    ax.annotate(f'Annual Cycle\n({periods[annual_idx]:.1f} days)',
                 xy=(periods[annual_idx], amplitudes[annual_idx]),
                 xytext=(periods[annual_idx] + 30, amplitudes[annual_idx] * 1.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    ax.annotate(f'Semi-Annual Cycle\n({periods[semi_annual_idx]:.1f} days)',
                 xy=(periods[semi_annual_idx], amplitudes[semi_annual_idx]),
                 xytext=(periods[semi_annual_idx] + 30, amplitudes[semi_annual_idx] * 1.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

    # Add grid and limit x-axis to meaningful periods.
    ax.grid(True)
    ax.set_xlim(0, 500)  # up to about 1.5 years

    # Save figure.
    file_name = f'spectral_analysis_{column}.png'
    data_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(data_folder / file_name, dpi=300, bbox_inches='tight')
    print(f"Saved spectral analysis plot to {(data_folder / file_name).as_posix()}")

    # Show plot.
    plt.show()

    # Return dominant periods.
    top_periods = periods[np.argsort(amplitudes)[-5:]]
    print(f"Top 5 dominant periods for {column}: {top_periods} days")
    return top_periods


def plot_ecm_adjustment(
        df,
        data_folder,
        error_correction_series,
        title='Error Correction Model Adjustment Path',
        commodity='corn',
    ):
    """Plot the error correction model adjustment path."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-len(error_correction_series):], error_correction_series)
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Deviation from Long-term Equilibrium')
    ax.set_title(title)
    ax.grid(True)

    # Save figure
    file_name = f'{commodity}_ecm_adjustment.png'
    data_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(data_folder / file_name, dpi=300, bbox_inches='tight')

    print(f"Saved ECM adjustment plot to {(data_folder / file_name).as_posix()}")

    plt.show()