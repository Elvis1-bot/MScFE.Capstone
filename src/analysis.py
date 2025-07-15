import warnings
import numpy as np # type: ignore
import pandas as pd # type: ignore
from statsmodels.tsa.stattools import adfuller # type: ignore
from statsmodels.tsa.arima.model import ARIMA # type: ignore
from statsmodels.tsa.vector_ar.vecm import coint_johansen # type: ignore
from statsmodels.tsa.vector_ar.var_model import VAR # type: ignore
import statsmodels.api as sm # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore
# from sklearn.model_selection import TimeSeriesSplit # type: ignore

warnings.filterwarnings("ignore")


def test_stationarity(series):
    """Test stationarity of a time series using Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna())

    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')

    if result[1] <= 0.05:
        print(f"Series is stationary (p-value <= 0.05)")
        return True
    else:
        print(f"Series is non-stationary (p-value > 0.05)")
        return False
    

def identify_arima_orders(series, max_p=5, max_d=2, max_q=5):
    """Identify optimal ARIMA orders using AIC."""
    # Make sure the series is stationary
    is_stationary = test_stationarity(series)
    
    if not is_stationary:
        print("Series is not stationary, differencing will be applied")
    
    best_aic = float('inf')
    best_order = None
    best_model = None
    
    # Try different combinations
    for p in range(max_p + 1):
        for d in range(1, max_d + 1) if not is_stationary else [0]:
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series.dropna(), order=(p, d, q))
                    model_fit = model.fit()
                    current_aic = model_fit.aic
                    
                    if current_aic < best_aic:
                        best_aic = current_aic
                        best_order = (p, d, q)
                        best_model = model_fit
                        
                except Exception as e:
                    continue
    
    print(f"Best ARIMA order: {best_order} with AIC: {best_aic:.4f}")
    return best_order, best_model

def fit_arima_model(series, order=None):
    """Fit an ARIMA model to the time series."""
    if order is None:
        order, _ = identify_arima_orders(series)
    
    model = ARIMA(series.dropna(), order=order)
    model_fit = model.fit()
    
    print(f"ARIMA Model Summary for {series.name}:")
    print(model_fit.summary())
    
    return model_fit


def johansen_cointegration_test(df, columns, significance_level=0.05):
    """Perform Johansen cointegration test and return rank and result object."""
    data = df[columns].dropna()
    result = coint_johansen(data, det_order=0, k_ar_diff=1)

    # Trace statistics and critical values
    trace_stat = result.lr1
    trace_crit = result.cvt  # columns: 10%, 5%, 1%

    # Identify index for chosen significance level
    levels = np.array([0.10, 0.05, 0.01])
    sig_idx = list(levels).index(significance_level)
    rank = int((trace_stat > trace_crit[:, sig_idx]).sum())

    print(f"Johansen Cointegration Test for {columns}:")
    print("Trace Statistics:\n", trace_stat)
    print(f"Critical Values ({significance_level*100:.0f}%):", trace_crit[:, sig_idx])
    print(f"Cointegration Rank: {rank}\n")

    return rank, result


def error_correction_model(df, columns, cointegration_rank=None, johansen_results=None):
    """Fit a vector error correction model and compute the error correction term."""
    data = df[columns].dropna()

    # Determine cointegration rank if not provided
    if cointegration_rank is None:
        cointegration_rank, johansen_results = johansen_cointegration_test(df, columns)

    # Fit VAR on differenced data
    diff_data = data.diff().dropna()
    var_model = VAR(diff_data)
    var_results = var_model.fit()

    print("Vector Error Correction Model (VAR on diffs) Summary:")
    print(var_results.summary())

    ec_series = None
    if cointegration_rank > 0:
        # Select the first cointegration vector if multiple
        beta = johansen_results.evec[:, :cointegration_rank]  # type: ignore
        if cointegration_rank > 1:
            print("Multiple cointegration vectors detected; using the first for ECM term.")
        vec = beta[:, 0]  # first eigenvector

        # Compute single error correction term: beta' * data'
        ec_term = data.values.dot(vec)
        ec_series = pd.Series(ec_term, index=data.index, name='ec_term')

        print(f"Error Correction Term (first vector)\n Mean: {ec_series.mean():.4f}, Std: {ec_series.std():.4f}\n")
    else:
        print("No cointegration relationship found; ECM term not computed.\n")

    return var_results, ec_series


def regression_model(df, dependent_var, independent_vars, lags=3):
    """Fit a regression model with lagged variables."""
    # Create a copy of the dataframe to avoid modifying the original
    reg_df = df[[dependent_var] + independent_vars].copy()
    
    # Add lagged variables
    if lags > 0:
        for var in independent_vars:
            for lag in range(1, lags + 1):
                reg_df[f"{var}_lag{lag}"] = reg_df[var].shift(lag)
    
    # Drop missing values
    reg_df = reg_df.dropna()
    
    # Prepare X and y
    X_cols = [col for col in reg_df.columns if col != dependent_var]
    X = reg_df[X_cols]
    y = reg_df[dependent_var]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X)
    results = model.fit()
    
    print(f"Regression Model for {dependent_var}:")
    print(results.summary())
    
    # Create a table of coefficients
    coef_df = pd.DataFrame({
        'Variable': results.params.index,
        'Coefficient': results.params.values,
        'Std Error': results.bse.values,
        'P-value': results.pvalues.values
    })
    
    print("\nCoefficient Table:")
    print(coef_df)
    
    return results, coef_df


def create_regression_table(results, data_folder, filename='corn_regression_results.csv'):
    """Create a CSV table of regression results."""
    # Extract key statistics
    coef = results.params
    std_err = results.bse
    p_values = results.pvalues
    conf_int = results.conf_int()
    
    # Create DataFrame
    table = pd.DataFrame({
        'Coefficient': coef,
        'Std Error': std_err,
        'P-value': p_values,
        'Lower CI': conf_int[0],
        'Upper CI': conf_int[1]
    })
    
    # Save to CSV
    data_folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(data_folder / filename, index_label='Variable')
    
    print(f"Saved regression results to {(data_folder / filename).as_posix()}")
    return table


def rolling_window_validation(
        df, data_folder, dependent_var, independent_vars,
        filename='corn_validation_results.csv', window_size=5*365, step=365
):
    """Perform rolling window cross-validation."""
    # Calculate number of windows
    total_periods = len(df)
    n_windows = (total_periods - window_size) // step

    if n_windows <= 0:
        print("Not enough data for rolling window validation")
        return None

    results = []

    for i in range(n_windows):
        start_idx = i * step
        train_end_idx = start_idx + window_size
        test_end_idx = min(train_end_idx + step, total_periods)
        
        # Split data
        train_data = df.iloc[start_idx:train_end_idx].copy() # Create a copy to avoid modifying original df
        test_data = df.iloc[train_end_idx:test_end_idx].copy()  # Create a copy

        # Handle missing and infinite values in training data
        train_X = train_data[independent_vars].replace([np.inf, -np.inf], np.nan).dropna()
        train_y = train_data[dependent_var].loc[train_X.index] # Keep only non-NaN y values.

        # Check if train_X is empty after dropping NaNs
        if train_X.empty:
            print(f"Warning: No valid data in training window {i+1}. Skipping.")
            continue  # Skip to the next iteration

        train_X = sm.add_constant(train_X)
        
        model = sm.OLS(train_y, train_X)
        results_train = model.fit()
        
        # Predict on test data
        test_X = test_data[independent_vars].replace([np.inf, -np.inf], np.nan) # Handle infs in test
        test_X = test_X.dropna() # Drop NaNs in test.  Crucially, do NOT drop rows from test_y here.
        test_y = test_data[dependent_var].loc[test_X.index] # Keep only non-NaN y values.
        test_X = sm.add_constant(test_X)
        
        # Check if test_X is empty after dropping NaNs
        if test_X.empty:
            print(f"Warning: No valid data in testing window {i+1}. Skipping prediction.")
            predictions = pd.Series([np.nan] * len(test_y), index=test_y.index)  # Create a series of NaNs
        else:
            predictions = results_train.predict(test_X)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        mae = mean_absolute_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)
        
        window_result = {
            'Window': i + 1,
            'Train Start': train_data.index[0],
            'Train End': train_data.index[-1],
            'Test Start': test_data.index[0],
            'Test End': test_data.index[-1],
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        
        results.append(window_result)
    
    # Combine results
    validation_df = pd.DataFrame(results)
    
    print("\nRolling Window Validation Results:")
    print(validation_df)
    
    # Calculate average metrics
    avg_metrics = {
        'RMSE': validation_df['RMSE'].mean(),
        'MAE': validation_df['MAE'].mean(),
        'R2': validation_df['R2'].mean()
    }
    
    print("\nAverage Validation Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save to CSV
    data_folder.mkdir(parents=True, exist_ok=True)
    validation_df.to_csv(data_folder / filename, index=False)
    
    return validation_df, avg_metrics
