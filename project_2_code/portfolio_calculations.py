import os
import math
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from datetime import timedelta
from scipy.stats import gamma
import scipy.special as sp
import numpy as np
import data_handling
from dotenv import load_dotenv
import logging
import csv

# Logging setup
load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script's directory
parent_dir = os.path.dirname(script_dir)

# Check calculations
CHECK = True
def calculate_excess_log_returns_from_prices(portfolio_spec,
                                             stock_prices_df,
                                             risk_free_rate_df):
    logger.info("Calculating excess log returns.")

    # Calculate the log returns for each stock
    stock_log_returns_df = np.log(stock_prices_df / stock_prices_df.shift(1))

    # Calculate the actual frequency of stock log returns in days
    dates_diff = stock_prices_df.index.to_series().diff().dt.days.dropna()
    average_frequency_stock_log_returns = dates_diff.mean()

    # Ensure the max diff is not significantly greater than the average
    assert dates_diff.max() <= average_frequency_stock_log_returns + 4, "Unexpected large gap between return dates."

    # Adjust the risk-free rate for the observed frequency
    # Assuming risk_free_rate_df contains annualized risk-free rates
    risk_free_rate_adjusted = (1 + risk_free_rate_df) ** (average_frequency_stock_log_returns / 365) - 1

    # Ensure risk_free_rate_adjusted index aligns with risk_free_rate_df for correct operations
    risk_free_rate_adjusted.index = risk_free_rate_df.index

    # Resample and interpolate risk-free rates to match stock returns' dates
    risk_free_rate_resampled = risk_free_rate_adjusted.reindex(stock_log_returns_df.index, method = 'ffill')

    # Calculate the excess log returns
    stock_excess_log_returns_df = stock_log_returns_df - risk_free_rate_resampled.values

    # Drop NaN values, which occur for the first data point
    stock_excess_log_returns_df.dropna(inplace = True)

    return stock_excess_log_returns_df

# def calculate_portfolio_variance(portfolio_weights_df,
#                                  covariance_matrix_df):
#     logger.info(f"Calculating portfolio variance.")

#     # Sort the portfolio DataFrame by index (stock symbols)
#     sorted_portfolio_weights_df = portfolio_weights_df.sort_index()
#     sorted_weights_np = sorted_portfolio_weights_df['Weight'].to_numpy()

#     # Sort the covariance DataFrame by stock symbols and convert to a numpy array
#     sorted_keys = sorted_portfolio_weights_df.index
#     sorted_covariance_matrix_df = covariance_matrix_df.loc[sorted_keys, sorted_keys]
#     sorted_covariance_matrix_np = sorted_covariance_matrix_df.to_numpy()

#     # Compute the portfolio variance as w^T * S * w
#     portfolio_variance = np.dot(sorted_weights_np.T, np.dot(sorted_covariance_matrix_np, sorted_weights_np))

#     logger.info(f"Portfolio Variance: {portfolio_variance}")

#     # Check calculations
#     if CHECK:
#         portfolio_variance_check = portfolio_weights_df["Weight"].T.dot(covariance_matrix_df.dot(portfolio_weights_df["Weight"]))
#         logger.info(f"Portfolio Variance Check: {portfolio_variance_check}")

#         is_close = np.isclose(portfolio_variance, portfolio_variance_check, atol = 1e-4)
#         if not is_close:
#             raise ValueError(f"Portfolio variance is not consistent.")

#     return portfolio_variance

def calculate_portfolio_variance(portfolio_weights_df, covariance_matrix_df):
    logger.info("Calculating portfolio variance.")

    # Sort weights and covariance matrix by asset names
    sorted_portfolio_weights_df = portfolio_weights_df.sort_index()
    sorted_weights_np = sorted_portfolio_weights_df['Weight'].to_numpy()
    sorted_keys = sorted_portfolio_weights_df.index
    sorted_covariance_matrix_df = covariance_matrix_df.loc[sorted_keys, sorted_keys]
    sorted_covariance_matrix_np = sorted_covariance_matrix_df.to_numpy()

    # Compute portfolio variance as w^T * S * w
    portfolio_variance = np.dot(sorted_weights_np.T, np.dot(sorted_covariance_matrix_np, sorted_weights_np))
    logger.info(f"Portfolio Variance: {portfolio_variance}")

    if CHECK:
        portfolio_variance_check = sorted_portfolio_weights_df["Weight"].T.dot(
            sorted_covariance_matrix_df.dot(sorted_portfolio_weights_df["Weight"])
        )
        logger.info(f"Portfolio Variance Check: {portfolio_variance_check}")
        # If both are NaN, treat them as consistent; otherwise, check with isclose.
        if (np.isnan(portfolio_variance) and np.isnan(portfolio_variance_check)):
            logger.warning("Both portfolio variance and its check are NaN.")
        elif not np.isclose(portfolio_variance, portfolio_variance_check, atol=1e-3):
            raise ValueError("Portfolio variance is not consistent.")
    return portfolio_variance

def calculate_average_mcm_window(portfolio_spec,
                                trading_date_ts,
                                mcm_prices_df):

    # Ensure k_stock_prices_df.index is a DateTimeIndex and is sorted
    mcm_prices_df = mcm_prices_df.sort_index()

    # Check if trading_date_ts is the last date in mcm_prices_df
    if trading_date_ts != mcm_prices_df.index[-1]:
        logger.error(f"trading_date_ts {trading_date_ts} is not the last date in the DataFrame.")
        raise ValueError(f"trading_date_ts {trading_date_ts} must be the last date in the DataFrame.")

    if portfolio_spec["rolling_window_frequency"] == "daily":
        mcm_prices_window_df = mcm_prices_df
    elif portfolio_spec["rolling_window_frequency"] == "weekly":
        # Resample to get the last trading day of each week
        mcm_prices_window_df = mcm_prices_df.resample('W').last()
    elif portfolio_spec["rolling_window_frequency"] == "monthly":
        # Resample to get the last trading day of each month
        mcm_prices_window_df = mcm_prices_df.resample('M').last()

    # Calculate the average mcm price over the specified rolling window
    average_mcm_price = mcm_prices_window_df.iloc[-portfolio_spec["rolling_window"]:].mean().item()

    return average_mcm_price

def get_window_annualization_factor(portfolio_spec):
    if portfolio_spec["rolling_window_frequency"] == "daily":
        window_annualization_factor= 252
    elif portfolio_spec["rolling_window_frequency"] == "weekly":
        window_annualization_factor = 52
    elif portfolio_spec["rolling_window_frequency"] == "monthly":
        window_annualization_factor = 12

    return window_annualization_factor

def get_window_trading_days(portfolio_spec):
    if portfolio_spec["rolling_window_frequency"] == "daily":
        window_days = portfolio_spec["rolling_window"]
    elif portfolio_spec["rolling_window_frequency"] == "weekly":
        window_days = portfolio_spec["rolling_window"]*5
    elif portfolio_spec["rolling_window_frequency"] == "monthly":
        window_days = portfolio_spec["rolling_window"]*22

    return window_days

def adjust_stock_prices_window(portfolio_spec,
                               trading_date_ts,
                               k_stock_prices_df):
    logger.info("Adjusting daily prices to rolling window.")

    # Ensure k_stock_prices_df.index is a DateTimeIndex and is sorted
    k_stock_prices_df = k_stock_prices_df.sort_index()

    # Check if trading_date_ts is the last date in k_stock_prices_df
    if trading_date_ts != k_stock_prices_df.index[-1]:
        logger.error(f"trading_date_ts {trading_date_ts} is not the last date in the DataFrame.")
        raise ValueError(f"trading_date_ts {trading_date_ts} must be the last date in the DataFrame.")

    if portfolio_spec["rolling_window_frequency"] == "daily":
        k_stock_prices_window_df = k_stock_prices_df
    elif portfolio_spec["rolling_window_frequency"] == "weekly":
        # Resample to get the last trading day of each week
        k_stock_prices_window_df = k_stock_prices_df.resample('W').last()
    elif portfolio_spec["rolling_window_frequency"] == "monthly":
        # Resample to get the last trading day of each month
        k_stock_prices_window_df = k_stock_prices_df.resample('M').last()

    # Select the last 'rolling_window' observations from the DataFrame
    k_stock_prices_window_df = k_stock_prices_window_df.iloc[-portfolio_spec["rolling_window"]:]

    return k_stock_prices_window_df

def calculate_canonical_statistics_T(portfolio_spec,
                                     trading_date_ts,
                                     k_stock_prices_df,
                                     risk_free_rate_df):
    logger.info(f"Calculating canonical statistics T.")

    # Adjust the stock prices DataFrame based on rolling window
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df
    )

    # Calculate Canonical Statistics T using DataFrame's dot product method
    canonical_statistics_T_df = k_stock_excess_log_returns_window_df.T.dot(
        k_stock_excess_log_returns_window_df
    )

    # Check calculations
    if CHECK:
        num_columns = k_stock_excess_log_returns_window_df.shape[1]
        canonical_statistics_T_matrix_check = np.zeros((num_columns, num_columns))
        for index, row in k_stock_excess_log_returns_window_df.iterrows():
            xi = row.values
            canonical_statistics_T_matrix_check += np.outer(xi, xi)

        # Convert the numpy matrix to a DataFrame
        canonical_statistics_T_df_check = pd.DataFrame(canonical_statistics_T_matrix_check)

        # Set the row and column labels to match those from k_stock_excess_log_returns_window_df
        canonical_statistics_T_df_check.columns = k_stock_excess_log_returns_window_df.columns
        canonical_statistics_T_df_check.index = k_stock_excess_log_returns_window_df.columns

        are_equal = np.isclose(canonical_statistics_T_df, canonical_statistics_T_df_check, rtol = 1e-4,
                                      atol = 1e-4).all().all()
        if not are_equal:
            raise ValueError(f"Canonical statistics T is not consistent.")

    return canonical_statistics_T_df

def calculate_canonical_statistics_t(portfolio_spec,
                                     trading_date_ts,
                                     k_stock_prices_df,
                                     risk_free_rate_df):
    logger.info(f"Calculating canonical statistics t.")

    # Adjust the stock prices DataFrame based on rolling window
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df)

    # Calculate Canonical Statistics t using DataFrame's sum method
    canonical_statistics_t_df = k_stock_excess_log_returns_window_df.sum(axis=0).to_frame()

    # Check calculations
    if CHECK:
        num_columns = k_stock_excess_log_returns_window_df.shape[1]
        canonical_statistics_t_vector_check = np.zeros(num_columns)
        for index, row in k_stock_excess_log_returns_window_df.iterrows():
            xi = row.values
            canonical_statistics_t_vector_check += xi

        # Convert the numpy vector to a DataFrame
        canonical_statistics_t_df_check = pd.DataFrame(canonical_statistics_t_vector_check)

        # Set the row labels to match those from k_stock_excess_log_returns_window_df
        canonical_statistics_t_df_check.index = k_stock_excess_log_returns_window_df.columns

        are_equal = np.isclose(canonical_statistics_t_df, canonical_statistics_t_df_check, rtol = 1e-4,
                                      atol = 1e-4).all().all()

        if not are_equal:
            raise ValueError(f"Canonical statistics t is not consistent.")


    return canonical_statistics_t_df

### New
## Bayes Factor function - Theorem 3.1 ORIGINAL 
# def compute_bayes_factor(mcm_dict, posterior_nu_dict, conjugate_prior_n_dict, conjugate_posterior_n_dict, conjugate_prior_S_dict, conjugate_posterior_S_dict, conjugate_prior_w_dict, conjugate_c_dict):
#     """
#     Computes the Bayes Factor for multiple climate risk factor models.

#     Args:
#     - mcm_dict (dict): Dictionary of Market Condition Metrics (MCMs).
#     - posterior_nu_dict (dict): Posterior estimates of `ν_j` for each risk factor.
#     - conjugate_prior_n_dict (dict): Prior sample size `n_j` for each risk factor.
#     - conjugate_posterior_n_dict (dict): Posterior sample size `n_c,j` for each risk factor.
#     - conjugate_prior_S_dict (dict): Prior covariance matrix `S_j` for each risk factor.
#     - conjugate_posterior_S_dict (dict): Posterior covariance matrix `S_{c,j}` for each risk factor.
#     - conjugate_prior_w_dict (dict): Prior weight vector `w_0` for each risk factor.
#     - conjugate_c_dict (dict): Scaling factor `c_j` for each risk factor (precomputed).

#     Returns:
#     - bayes_factor_dict (dict): Bayes Factors for each risk factor.
#     """

#     logger.info("Computing exact Bayes Factors for all risk factors.")

#     bayes_factor_dict = {}

#     for mcm_name in mcm_dict.keys():
#         logger.info(f"Computing Bayes Factor for {mcm_name}")

#         # Retrieve parameters for the current MCM
#         p = len(posterior_nu_dict[mcm_name])  # Number of assets
#         n_j = conjugate_prior_n_dict[mcm_name]
#         n_cj = conjugate_posterior_n_dict[mcm_name]

#         # Convert covariance matrices to numpy arrays
#         S_prior = conjugate_prior_S_dict[mcm_name].values
#         S_post = conjugate_posterior_S_dict[mcm_name].values

#         # Convert weight vectors to column vectors (2D arrays)
#         w0_arr = conjugate_prior_w_dict[mcm_name]['Weight'].values.reshape(-1, 1)
#         nu_arr = posterior_nu_dict[mcm_name].values.reshape(-1, 1)

#         c_j = conjugate_c_dict[mcm_name]

#         # Compute determinant adjustments using the formula:
#         # det( S - (scaling term) * S @ w @ w.T @ S )
#         det_adjusted_Sj = np.linalg.det(S_prior - (c_j ** 2) * (n_j ** -1) * (S_prior @ w0_arr @ w0_arr.T @ S_prior))
#         det_adjusted_Scj = np.linalg.det(S_post - (n_cj ** -1) * (S_post @ nu_arr @ nu_arr.T @ S_post))

#         # Compute multivariate gamma ratio (using log values from multigammaln)
#         gamma_ratio = sp.multigammaln((n_cj + p + 2) / 2, p) - sp.multigammaln((n_j + p + 2) / 2, p)

#         # Compute the Bayes factor as per the paper
#         bayes_factor = (
#             (np.pi ** (-p * n_j / 2))
#             * ((n_j / n_cj) ** (p / 2))
#             * np.exp(gamma_ratio)
#             * (det_adjusted_Sj ** ((n_j + p + 2) / 2))
#             / (det_adjusted_Scj ** ((n_cj + p + 2) / 2))
#         )

#         # Handle numerical issues
#         if np.isnan(bayes_factor) or bayes_factor <= 0:
#             logger.warning(f"Bayes Factor for {mcm_name} is NaN or non-positive. Assigning default value.")
#             bayes_factor = 1e-6

#         bayes_factor_dict[mcm_name] = bayes_factor

#     return bayes_factor_dict

### Test - normal calculation
# def compute_bayes_factor(mcm_dict, posterior_nu_dict, conjugate_prior_n_dict, 
#                          conjugate_posterior_n_dict, conjugate_prior_S_dict, 
#                          conjugate_posterior_S_dict, conjugate_prior_w_dict, 
#                          conjugate_c_dict):
#     """
#     Computes the Bayes Factor for multiple climate risk factor models.

#     Args:
#       - mcm_dict (dict): Dictionary of Market Condition Metrics (MCMs).
#       - posterior_nu_dict (dict): Posterior estimates of `ν_j` for each risk factor.
#       - conjugate_prior_n_dict (dict): Prior sample size `n_j` for each risk factor.
#       - conjugate_posterior_n_dict (dict): Posterior sample size `n_c,j` for each risk factor.
#       - conjugate_prior_S_dict (dict): Prior covariance matrix `S_j` for each risk factor.
#       - conjugate_posterior_S_dict (dict): Posterior covariance matrix `S_{c,j}` for each risk factor.
#       - conjugate_prior_w_dict (dict): Prior weight vector `w_0` for each risk factor.
#       - conjugate_c_dict (dict): Scaling factor `c_j` for each risk factor (precomputed).

#     Returns:
#       - bayes_factor_dict (dict): Bayes Factors for each risk factor.
#     """
#     logger.info("Computing exact Bayes Factors for all risk factors.")
#     bayes_factor_dict = {}

#     for mcm_name in mcm_dict.keys():
#         logger.info(f"Computing Bayes Factor for {mcm_name}")

#         # Retrieve parameters for the current MCM
#         p = len(posterior_nu_dict[mcm_name])  # Number of assets
#         n_j = conjugate_prior_n_dict[mcm_name]
#         n_cj = conjugate_posterior_n_dict[mcm_name]

#         # Convert covariance matrices to NumPy arrays to avoid alignment issues
#         prior_Sj_arr = np.asarray(conjugate_prior_S_dict[mcm_name])
#         posterior_Scj_arr = np.asarray(conjugate_posterior_S_dict[mcm_name])
        
#         # Convert weight vectors and posterior_nu to 2D column vectors
#         prior_w0_vec = np.array(conjugate_prior_w_dict[mcm_name]['Weight'])
#         if prior_w0_vec.ndim == 1:
#             prior_w0_vec = prior_w0_vec.reshape(-1, 1)
#         posterior_nu_vec = np.array(posterior_nu_dict[mcm_name])
#         if posterior_nu_vec.ndim == 1:
#             posterior_nu_vec = posterior_nu_vec.reshape(-1, 1)
            
#         c_j = conjugate_c_dict[mcm_name]

#         # Print debugging information
#         print(f"\n🔎 Debugging Bayes Factor for {mcm_name}:")
#         print(f"p (num assets): {p}")
#         print(f"n_j (prior sample size): {n_j}")
#         print(f"n_cj (posterior sample size): {n_cj}")
#         print(f"c_j (scaling factor): {c_j}")

#         # Compute determinant adjustments for S_j and S_c,j
#         try:
#             term1 = prior_Sj_arr - (c_j ** 2) * (n_j ** -1) * (prior_Sj_arr @ prior_w0_vec @ prior_w0_vec.T @ prior_Sj_arr)
#             det_adjusted_Sj = np.linalg.det(term1)
#             term2 = posterior_Scj_arr - (n_cj ** -1) * (posterior_Scj_arr @ posterior_nu_vec @ posterior_nu_vec.T @ posterior_Scj_arr)
#             det_adjusted_Scj = np.linalg.det(term2)

#             print(f"det_adjusted_Sj: {det_adjusted_Sj}")
#             print(f"det_adjusted_Scj: {det_adjusted_Scj}")

#         except np.linalg.LinAlgError as e:
#             print(f" ERROR computing determinant for {mcm_name}: {e}")
#             det_adjusted_Sj = np.nan
#             det_adjusted_Scj = np.nan

#         # Compute multivariate gamma ratio
#         try:
#             gamma_ratio = sp.multigammaln((n_cj + p + 2) / 2, p) - sp.multigammaln((n_j + p + 2) / 2, p)
#             print(f"gamma_ratio: {gamma_ratio}")
#         except ValueError as e:
#             print(f" ERROR computing gamma ratio for {mcm_name}: {e}")
#             gamma_ratio = np.nan

#         # Compute the full Bayes factor equation
#         try:
#             bayes_factor = (
#                 (np.pi ** (-p * n_j / 2)) *
#                 ((n_j / n_cj) ** (p / 2)) *
#                 np.exp(gamma_ratio) *
#                 (det_adjusted_Sj ** ((n_j + p + 2) / 2)) /
#                 (det_adjusted_Scj ** ((n_cj + p + 2) / 2))
#             )
#             print(f"Bayes Factor: {bayes_factor}")
#         except (ValueError, OverflowError) as e:
#             print(f" ERROR computing Bayes Factor for {mcm_name}: {e}")
#             bayes_factor = np.nan

#         # Handle numerical stability issues
#         if np.isnan(bayes_factor) or bayes_factor <= 0:
#             logger.warning(f"Bayes Factor for {mcm_name} is NaN or non-positive. Assigning default value.")
#             bayes_factor = 1e-6

#         bayes_factor_dict[mcm_name] = bayes_factor

#     return bayes_factor_dict

def compute_bayes_factor(mcm_dict, posterior_nu_dict, conjugate_prior_n_dict, 
                         conjugate_posterior_n_dict, conjugate_prior_S_dict, 
                         conjugate_posterior_S_dict, conjugate_prior_w_dict, 
                         conjugate_c_dict):
    """
    Computes relative Bayes Factors for each risk factor (model) using log transformations
    to ensure numerical stability. The returned values are normalized to sum to one.
    
    Args:
      - mcm_dict (dict): Dictionary of Market Condition Metrics (MCMs).
      - posterior_nu_dict (dict): Posterior estimates of `ν_j` for each risk factor.
      - conjugate_prior_n_dict (dict): Prior sample size `n_j` for each risk factor.
      - conjugate_posterior_n_dict (dict): Posterior sample size `n_{c,j}` for each risk factor.
      - conjugate_prior_S_dict (dict): Prior covariance matrix `S_j` for each risk factor.
      - conjugate_posterior_S_dict (dict): Posterior covariance matrix `S_{c,j}` for each risk factor.
      - conjugate_prior_w_dict (dict): Prior weight vector `w_0` for each risk factor.
      - conjugate_c_dict (dict): Scaling factor `c_j` for each risk factor.
      
    Returns:
      - normalized_bayes_factors (dict): Normalized Bayes Factors (model probabilities) for each risk factor.
    """
    logger.info("Computing exact Bayes Factors for all risk factors (log scale).")

    log_bayes_factors = {}
    bayes_factors_dict = {}

    for mcm_name in mcm_dict.keys():
        logger.info(f"Computing Bayes Factor for {mcm_name}")

        # Retrieve parameters
        p = len(posterior_nu_dict[mcm_name])  # number of assets
        n_j = conjugate_prior_n_dict[mcm_name]
        n_cj = conjugate_posterior_n_dict[mcm_name]

        # Convert covariance matrices to NumPy arrays
        prior_Sj_arr = np.asarray(conjugate_prior_S_dict[mcm_name])
        posterior_Scj_arr = np.asarray(conjugate_posterior_S_dict[mcm_name])
        
        # Ensure that the weight vector and posterior nu are 2D column vectors
        prior_w0_vec = np.array(conjugate_prior_w_dict[mcm_name]['Weight'])
        if prior_w0_vec.ndim == 1:
            prior_w0_vec = prior_w0_vec.reshape(-1, 1)
        posterior_nu_vec = np.array(posterior_nu_dict[mcm_name])
        if posterior_nu_vec.ndim == 1:
            posterior_nu_vec = posterior_nu_vec.reshape(-1, 1)
        c_j = conjugate_c_dict[mcm_name]

        # Debug prints
        print(f"\n🔎 Debugging Bayes Factor for {mcm_name}:")
        print(f"p (num assets): {p}")
        print(f"n_j (prior sample size): {n_j}")
        print(f"n_cj (posterior sample size): {n_cj}")
        print(f"c_j (scaling factor): {c_j}")

        try:
            term1 = prior_Sj_arr - (c_j ** 2) * (n_j ** -1) * (prior_Sj_arr @ prior_w0_vec @ prior_w0_vec.T @ prior_Sj_arr)
            term2 = posterior_Scj_arr - (n_cj ** -1) * (posterior_Scj_arr @ posterior_nu_vec @ posterior_nu_vec.T @ posterior_Scj_arr)
            det_adjusted_Sj = np.linalg.det(term1)
            det_adjusted_Scj = np.linalg.det(term2)
            print(f"det_adjusted_Sj: {det_adjusted_Sj}")
            print(f"det_adjusted_Scj: {det_adjusted_Scj}")
        except np.linalg.LinAlgError as e:
            print(f" ERROR computing determinant for {mcm_name}: {e}")
            continue

        try:
            gamma_ratio = sp.multigammaln((n_cj + p + 2) / 2, p) - sp.multigammaln((n_j + p + 2) / 2, p)
            print(f"gamma_ratio: {gamma_ratio}")
        except ValueError as e:
            print(f" ERROR computing gamma ratio for {mcm_name}: {e}")
            continue

        # Compute the log Bayes factor (to avoid overflow)
        try:
            log_bf = (-p * n_j / 2) * np.log(np.pi) + (p / 2) * np.log(n_j / n_cj) \
                     + gamma_ratio \
                     + ((n_j + p + 2) / 2) * np.log(det_adjusted_Sj) \
                     - ((n_cj + p + 2) / 2) * np.log(det_adjusted_Scj)
            log_bayes_factors[mcm_name] = log_bf
            print(f"Log Bayes Factor for {mcm_name}: {log_bf}")
        except Exception as e:
            print(f" ERROR computing log Bayes Factor for {mcm_name}: {e}")
            continue

    # Now, convert log Bayes factors into relative Bayes factors (model weights)
    if not log_bayes_factors:
        raise ValueError("No valid Bayes factors were computed.")
    max_log_bf = max(log_bayes_factors.values())
    bayes_factors = {mcm: np.exp(log_bayes_factors[mcm] - max_log_bf) for mcm in log_bayes_factors}
    total_bf = sum(bayes_factors.values())
    bayes_factors_dict = {mcm: bf / total_bf for mcm, bf in bayes_factors.items()}
    
    return bayes_factors_dict



### 
### Adjusted to allow several MCMs at a time. Arg difference: mcm_dict instead of mcm_prices_df
## Output is a dictionary with n_j for each MCM
def calculate_conjugate_prior_n(portfolio_spec,
                                trading_date_ts,
                                mcm_dict):
    logger.info(f"Calculating conjugate prior n for multiple risk factors.")

    conjugate_prior_n_dict = {}

    for mcm_name, mcm_prices_df in mcm_dict.items():
        average_mcm_price = calculate_average_mcm_window(portfolio_spec, trading_date_ts, mcm_prices_df)
        current_mcm_price = mcm_prices_df.loc[trading_date_ts].item()

        if current_mcm_price > average_mcm_price:
            mcm_price_fraction = (current_mcm_price / average_mcm_price)
        else:
            mcm_price_fraction = (average_mcm_price / current_mcm_price)

        conjugate_prior_n_dict[mcm_name] = portfolio_spec["rolling_window"] * mcm_price_fraction * portfolio_spec["mcm_scaling"]
        
    return conjugate_prior_n_dict


### Adjust the posterior sample size for each MCM from prior sample size
### Outputs n_j + n for all MCMs
def calculate_conjugate_posterior_n(portfolio_spec,
                                    trading_date_ts,
                                    mcm_dict,
                                    conjugate_prior_n_dict=None):
    logger.info(f"Calculating conjugate posterior n for multiple risk factors.")

    # If conjugate_prior_n is not provided, calculate it
    if conjugate_prior_n_dict is None:
        conjugate_prior_n_dict = calculate_conjugate_prior_n(portfolio_spec, trading_date_ts, mcm_dict)

    conjugate_posterior_n_dict = {mcm_name: n_j + portfolio_spec["rolling_window"] for mcm_name, n_j in conjugate_prior_n_dict.items()} 

    return conjugate_posterior_n_dict

## Outputs S_j for all MCMs
def calculate_conjugate_prior_S(portfolio_spec,
                                trading_date_ts,
                                k_stock_intraday_prices_df,
                                mcm_dict,
                                conjugate_prior_n_dict=None):
    logger.info(f"Calculating conjugate prior S.")

    # If conjugate_prior_n is not provided, calculate it
    if conjugate_prior_n_dict is None:
        conjugate_prior_n_dict = calculate_conjugate_prior_n(portfolio_spec, trading_date_ts, mcm_dict)  

    # Calculate log returns for the last period of intraday prices
    if portfolio_spec["rolling_window_frequency"] == "daily":
        days_in_single_rolling_window = 1
    elif portfolio_spec["rolling_window_frequency"] == "weekly":
        days_in_single_rolling_window = 7
    elif portfolio_spec["rolling_window_frequency"] == "monthly":
        days_in_single_rolling_window = 31
    else:
        # Log an error and raise an exception if the rebalancing frequency is unknown
        logger.error("Unknown rolling window frequency.")
        raise RuntimeError("Unknown rolling window frequency.")

    conjugate_prior_S_dict = {}

    for mcm_name, n_j in conjugate_prior_n_dict.items():
        hf_start_date = trading_date_ts - pd.Timedelta(days = days_in_single_rolling_window)
        filtered_intraday_prices_df = k_stock_intraday_prices_df.loc[hf_start_date:trading_date_ts]
    
        log_returns = np.log(filtered_intraday_prices_df / filtered_intraday_prices_df.shift(1)).dropna()
        realized_cov_matrix = log_returns.cov() * len(log_returns)

        # Check calculations
        # Maybe remove this check
        if CHECK:
            if not np.allclose(realized_cov_matrix, realized_cov_matrix.T, rtol=1e-3, atol=1e-3):
                raise ValueError("Realized covariance matrix is not symmetric.")
            
        conjugate_prior_S_dict[mcm_name] = n_j * realized_cov_matrix

    # Return the dictionary of scaled covariance matrix
    return conjugate_prior_S_dict

## Outputs S_j + T for all MCMs
def calculate_conjugate_posterior_S(portfolio_spec,
                                    trading_date_ts,
                                    k_stock_prices_df,
                                    k_stock_intraday_prices_df,
                                    mcm_dict,
                                    risk_free_rate_df,
                                    conjugate_prior_S_dict=None):
    logger.info(f"Calculating conjugate posterior S for multiple risk factors.")

    # If conjugate_prior_S_df is not provided, calculate it
    if conjugate_prior_S_dict is None:
        conjugate_prior_S_dict = calculate_conjugate_prior_S(portfolio_spec, trading_date_ts, k_stock_intraday_prices_df, mcm_dict)


    # Calculate the Canonical Statistics T matrix
    canonical_statistics_T_df = calculate_canonical_statistics_T(portfolio_spec,
                                                              trading_date_ts,
                                                              k_stock_prices_df,
                                                              risk_free_rate_df)

    conjugate_posterior_S_dict = {mcm_name: S_j + canonical_statistics_T_df for mcm_name, S_j in conjugate_prior_S_dict.items()}

    # Return the sum of conjugate_prior_S_df and canonical_statistics_T
    return conjugate_posterior_S_dict


## Outputs w_j for all MCMs
def calculate_conjugate_prior_w(portfolio_spec,
                                trading_date_ts,
                                k_stock_prices_df,
                                k_stock_market_caps_df,
                                mcm_dict):
    logger.info("Calculating conjugate prior w.")

    conjugate_prior_w_dict = {}

    # If the weighting strategy includes 'vw' or if it is one of the conjugate strategies,
    # default to using the value-weighted portfolio.
    if "vw" in portfolio_spec["weighting_strategy"] or "conjugate" in portfolio_spec["weighting_strategy"]:
        portfolio_weights_df = calculate_value_weighted_portfolio(portfolio_spec, trading_date_ts, k_stock_market_caps_df)
    elif "ew" in portfolio_spec["weighting_strategy"]:
        portfolio_weights_df = calculate_equally_weighted_portfolio(portfolio_spec, k_stock_prices_df)
    else:
        logger.error("Unknown conjugate portfolio prior weights.")
        raise ValueError("Unknown conjugate portfolio prior weights.")

    # Assign the same prior weights for each MCM.
    for mcm_name in mcm_dict.keys():
        conjugate_prior_w_dict[mcm_name] = portfolio_weights_df

    return conjugate_prior_w_dict

# Output c_j for all MCMs
def calculate_conjugate_c(portfolio_spec, 
                          trading_date_ts, 
                          k_stock_prices_df, 
                          k_stock_market_caps_df, 
                          k_stock_intraday_prices_df, 
                          mcm_dict, 
                          conjugate_prior_n_dict=None, 
                          conjugate_prior_S_dict=None, 
                          conjugate_prior_w_dict=None):
    
    logger.info(f"Calculating conjugate c for multiple risk factors.")

    if conjugate_prior_n_dict is None:
        conjugate_prior_n_dict = calculate_conjugate_prior_n(portfolio_spec, trading_date_ts, mcm_dict)

    if conjugate_prior_S_dict is None:
        conjugate_prior_S_dict = calculate_conjugate_prior_S(portfolio_spec, trading_date_ts, k_stock_intraday_prices_df, mcm_dict)

    if conjugate_prior_w_dict is None:
        conjugate_prior_w_dict = calculate_conjugate_prior_w(portfolio_spec, trading_date_ts, k_stock_prices_df, k_stock_market_caps_df, mcm_dict)

    conjugate_c_dict = {}
    
    # Equation 17 in the paper
    for mcm_name in mcm_dict.keys():
        c_j = (2 * conjugate_prior_n_dict[mcm_name]) / (
            (conjugate_prior_n_dict[mcm_name] + portfolio_spec["size"] + 2) +
            ((conjugate_prior_n_dict[mcm_name] + portfolio_spec["size"] + 2) ** 2 +
             4 * conjugate_prior_n_dict[mcm_name] * calculate_portfolio_variance(conjugate_prior_w_dict[mcm_name], conjugate_prior_S_dict[mcm_name])
            ) ** (1 / 2)
        )
        conjugate_c_dict[mcm_name] = c_j

    return conjugate_c_dict


# Calculate the posterior portfolio weights w_TP_j


def calculate_conjugate_posterior_w(portfolio_spec,
                                    trading_date_ts,
                                    k_stock_prices_df,
                                    k_stock_market_caps_df,
                                    k_stock_intraday_prices_df,
                                    mcm_dict,
                                    risk_free_rate_df,
                                    conjugate_c_dict=None,
                                    conjugate_prior_w_dict=None,
                                    conjugate_prior_S_dict=None,
                                    conjugate_posterior_S_dict=None):
    logger.info(f"Calculating conjugate posterior w with Bayesian Model Averaging.")

    # Compute priors if missing
    if conjugate_prior_w_dict is None:
        conjugate_prior_w_dict = calculate_conjugate_prior_w(portfolio_spec, trading_date_ts, k_stock_prices_df, k_stock_market_caps_df, mcm_dict)
    if conjugate_prior_S_dict is None:
        conjugate_prior_S_dict = calculate_conjugate_prior_S(portfolio_spec, trading_date_ts, k_stock_intraday_prices_df, mcm_dict)
    if conjugate_posterior_S_dict is None:
        conjugate_posterior_S_dict = calculate_conjugate_posterior_S(portfolio_spec, trading_date_ts, k_stock_prices_df, k_stock_intraday_prices_df, mcm_dict, risk_free_rate_df)
    if conjugate_c_dict is None:
        conjugate_c_dict = calculate_conjugate_c(portfolio_spec, trading_date_ts, k_stock_prices_df, k_stock_market_caps_df, k_stock_intraday_prices_df, mcm_dict)

    # Compute aggregated ν using Bayesian Model Averaging
    aggregated_nu = calculate_mean_conjugate_posterior_nu(portfolio_spec,
                                          trading_date_ts,
                                          k_stock_prices_df,
                                          k_stock_market_caps_df,
                                          k_stock_intraday_prices_df,
                                          mcm_dict,
                                          risk_free_rate_df,
                                          conjugate_c_dict=None,
                                          conjugate_prior_n_dict=None,
                                          conjugate_posterior_n_dict=None,
                                          conjugate_prior_S_dict=None,
                                          conjugate_posterior_S_dict=None,
                                          conjugate_prior_w_dict=None,
                                          conjugate_posterior_w_dict=None)


    # Compute posterior portfolio weights using Equation (19)
    posterior_weights = (1 / portfolio_spec["risk_aversion"]) * aggregated_nu

    # Convert to DataFrame with stock labels
    posterior_weights_df = pd.DataFrame(posterior_weights, columns=['Weight'])
    posterior_weights_df.index = k_stock_prices_df.columns  # Ensure correct stock indexing

    return posterior_weights_df


def calculate_mean_conjugate_posterior_nu(portfolio_spec,
                                          trading_date_ts,
                                          k_stock_prices_df,
                                          k_stock_market_caps_df,
                                          k_stock_intraday_prices_df,
                                          mcm_dict,
                                          risk_free_rate_df,
                                          conjugate_c_dict=None,
                                          conjugate_prior_n_dict=None,
                                          conjugate_posterior_n_dict=None,
                                          conjugate_prior_S_dict=None,
                                          conjugate_posterior_S_dict=None,
                                          conjugate_prior_w_dict=None,
                                          conjugate_posterior_w_dict=None):
    logger.info(f"Calculating mean conjugate posterior nu")

    # Compute shared conjugate parameters if missing:
    if conjugate_c_dict is None:
        conjugate_c_dict = calculate_conjugate_c(portfolio_spec, trading_date_ts, k_stock_prices_df,
                                                 k_stock_market_caps_df, k_stock_intraday_prices_df, mcm_dict)
    if conjugate_posterior_n_dict is None:
        conjugate_posterior_n_dict = calculate_conjugate_posterior_n(portfolio_spec, trading_date_ts, mcm_dict)
    if conjugate_posterior_S_dict is None:
        conjugate_posterior_S_dict = calculate_conjugate_posterior_S(portfolio_spec, trading_date_ts,
                                                                     k_stock_prices_df, k_stock_intraday_prices_df,
                                                                     mcm_dict, risk_free_rate_df)
    # *** Do NOT call calculate_conjugate_posterior_w here ***
    if conjugate_posterior_w_dict is None:
    # Use the prior weights as a fallback for the posterior weights
        conjugate_posterior_w_dict = calculate_conjugate_prior_w(portfolio_spec, trading_date_ts, k_stock_prices_df,
                                                             k_stock_market_caps_df, mcm_dict)


    # Also compute the prior dictionaries if missing
    if conjugate_prior_n_dict is None:
        conjugate_prior_n_dict = calculate_conjugate_prior_n(portfolio_spec, trading_date_ts, mcm_dict)
    if conjugate_prior_S_dict is None:
        conjugate_prior_S_dict = calculate_conjugate_prior_S(portfolio_spec, trading_date_ts, k_stock_intraday_prices_df, mcm_dict)
    if conjugate_prior_w_dict is None:
        conjugate_prior_w_dict = calculate_conjugate_prior_w(portfolio_spec, trading_date_ts, k_stock_prices_df,
                                                             k_stock_market_caps_df, mcm_dict)

    posterior_nu_dict = {}

    for mcm_name in mcm_dict.keys():
        posterior_nu_j = (conjugate_posterior_n_dict[mcm_name] + portfolio_spec["size"] + 2) * \
                         conjugate_posterior_w_dict[mcm_name] / \
                         (conjugate_posterior_n_dict[mcm_name] - 
                          calculate_portfolio_variance(conjugate_posterior_w_dict[mcm_name],
                                                       conjugate_posterior_S_dict[mcm_name]))
        posterior_nu_dict[mcm_name] = posterior_nu_j

    # Compute Bayes factors and perform BMA as before...
    bayes_factor_dict = compute_bayes_factor(
        mcm_dict,
        posterior_nu_dict,
        conjugate_prior_n_dict,
        conjugate_posterior_n_dict,
        conjugate_prior_S_dict,
        conjugate_posterior_S_dict,
        conjugate_prior_w_dict,
        conjugate_c_dict
    )

    total_bayes_factor = sum(bayes_factor_dict.values())
    if np.isnan(total_bayes_factor) or total_bayes_factor == 0:
        logger.warning("Total Bayes factor is NaN or zero. Applying uniform weighting.")
        bayes_weights = {mcm_name: 1 / len(mcm_dict) for mcm_name in mcm_dict.keys()}
    else:
        bayes_weights = {mcm_name: bayes_factor_dict[mcm_name] / total_bayes_factor for mcm_name in mcm_dict.keys()}

    aggregated_nu = sum(bayes_weights[mcm_name] * posterior_nu_dict[mcm_name] for mcm_name in mcm_dict.keys())

    return aggregated_nu



def calculate_mean_jeffreys_posterior_nu(portfolio_spec,
                                          trading_date_ts,
                                          k_stock_prices_df,
                                          risk_free_rate_df):
    # Log information about the calculation
    logger.info(f"Calculating mean Jeffreys posterior nu")

    # Calculate 'canonical_statistics_t'
    canonical_statistics_t_df = calculate_canonical_statistics_t(portfolio_spec,
                                                              trading_date_ts,
                                                              k_stock_prices_df,
                                                              risk_free_rate_df)
    # Calculate 'canonical_statistics_T'
    canonical_statistics_T_df = calculate_canonical_statistics_T(portfolio_spec,
                                                              trading_date_ts,
                                                              k_stock_prices_df,
                                                              risk_free_rate_df)


    # Compute and return the mean Jeffreys posterior nu
    jeffreys_scaled_covariance_matrix_df = (canonical_statistics_T_df - 1 / portfolio_spec["rolling_window"] * \
                                    canonical_statistics_t_df.dot(canonical_statistics_t_df.T))
    jeffreys_scaled_covariance_matrix_inv_df = pd.DataFrame(np.linalg.inv(jeffreys_scaled_covariance_matrix_df.values),
                                     index=jeffreys_scaled_covariance_matrix_df.columns,
                                     columns=jeffreys_scaled_covariance_matrix_df.index)

    mean_jeffreys_posterior_nu_df = jeffreys_scaled_covariance_matrix_inv_df.dot(canonical_statistics_t_df)
    mean_jeffreys_posterior_nu_df.columns = ['Weight']
    return mean_jeffreys_posterior_nu_df


def get_k_largest_stocks_market_caps(stock_market_caps_df,
                                     stock_prices_df,
                                     stock_intraday_prices_df,
                                     trading_date_ts,
                                     portfolio_size,
                                     rolling_window_days,
                                     rolling_window_frequency):
    # Get S&P 500 components for the current date
    tickers_list = data_handling.extract_unique_tickers(trading_date_ts,
                                          trading_date_ts)

    # Identify tickers that are present in stock_market_caps_df.columns
    present_tickers = [ticker for ticker in tickers_list if ticker in stock_market_caps_df.columns]
    missing_fraction = (len(tickers_list) - len(present_tickers)) / len(tickers_list)
    logger.info(f"Fraction of tickers missing from stock_market_caps_df: {missing_fraction:.2%}")

    # Days in single rolling window
    if rolling_window_frequency == "daily":
        days_in_single_rolling_window = 1
    elif rolling_window_frequency == "weekly":
        days_in_single_rolling_window = 7
    elif rolling_window_frequency == "monthly":
        days_in_single_rolling_window = 31
    else:
        # Log an error and raise an exception if the rebalancing frequency is unknown
        logger.error("Unknown rolling window frequency.")
        raise RuntimeError("Unknown rolling window frequency.")

    eligible_stocks = [
        stock for stock in stock_prices_df.columns
        if (
                stock in tickers_list and
                stock in stock_market_caps_df.columns and
                stock in stock_intraday_prices_df.columns and
                stock_prices_df.loc[trading_date_ts, stock] is not None and
                stock_prices_df[stock].loc[:trading_date_ts].tail(rolling_window_days).notna().all() and
                stock_intraday_prices_df[stock].loc[(trading_date_ts - timedelta(days=days_in_single_rolling_window)):(trading_date_ts + timedelta(days = 1))].notna().any()
        )
    ]

    # From these available stocks, get the size largest based on market caps for the current date
    if trading_date_ts in stock_market_caps_df.index:
        daily_market_caps = stock_market_caps_df.loc[trading_date_ts, eligible_stocks].dropna()
        k_stock_market_caps_df = daily_market_caps.nlargest(portfolio_size)
        return k_stock_market_caps_df
    else:
        logger.error(f"The trading date {trading_date_ts} does not exist in the market capitalizations data.")
        raise ValueError(f"The trading date {trading_date_ts} does not exist in the market capitalizations data.")


def calculate_equally_weighted_portfolio(portfolio_spec,
                                         k_stock_prices_df):
    # Logging the calculation step
    logger.info(f"Calculating equally weighted portfolio")

    # Determine the number of stocks in the portfolio
    num_stocks = portfolio_spec["size"]

    # Assign equal weight to each stock and create the resulting dataframe
    portfolio_weights_df = pd.DataFrame({
        'Weight': [1 / num_stocks] * num_stocks
    }, index=k_stock_prices_df.columns)

    # Rename the index to 'Stock'
    portfolio_weights_df.index.name = 'Stock'

    return portfolio_weights_df

def calculate_value_weighted_portfolio(portfolio_spec, 
                                       trading_date_ts, 
                                       k_stock_market_caps_df):
    logger.info(f"Calculating market cap portfolio weights.")
    k_stock_market_caps_series = k_stock_market_caps_df.iloc[-1].sort_values(ascending=False)

    # Extract the last index date from k_stock_market_caps_df
    last_index_date = k_stock_market_caps_df.index[-1]

    # Assert that the last index date is the same as trading_date_ts
    assert last_index_date == trading_date_ts, "The last index date does not match the trading date."

    # Total market cap of the k largest stocks
    total_market_cap = k_stock_market_caps_series.sum()

    # Calculate value weights
    portfolio_weights_df = pd.DataFrame(k_stock_market_caps_series / total_market_cap)

    # Fix labels
    portfolio_weights_df.index.name = 'Stock'
    portfolio_weights_df.columns = ['Weight']

    return portfolio_weights_df

def calculate_shrinkage_portfolio(portfolio_spec, 
                                  trading_date_ts, 
                                  k_stock_prices_df,
                                  risk_free_rate_df):
    logger.info(f"Calculating shrinkage portfolio weights.")
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec,
        trading_date_ts,
        k_stock_prices_df)

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df
    )

    window_annualization_factor = get_window_annualization_factor(portfolio_spec)

    # Mean return
    mean_log_returns = expected_returns.mean_historical_return(k_stock_excess_log_returns_window_df,
                                                                 returns_data=True,
                                                                 compounding=False,
                                                                  frequency = window_annualization_factor)

    # Covariance matrix
    covariance_log_returns = risk_models.CovarianceShrinkage(k_stock_excess_log_returns_window_df,
                                                                returns_data=True,
                                                                frequency = window_annualization_factor).ledoit_wolf()

    # Add risk free asset
    mean_log_returns_with_risk_free_asset = mean_log_returns.copy()
    mean_log_returns_with_risk_free_asset["RISK_FREE"] = 0

    covariance_log_returns_with_risk_free_asset = covariance_log_returns.copy()
    covariance_log_returns_with_risk_free_asset["RISK_FREE"] = 0
    covariance_log_returns_with_risk_free_asset.loc["RISK_FREE"] = 0

    ef = EfficientFrontier(mean_log_returns_with_risk_free_asset, covariance_log_returns_with_risk_free_asset, weight_bounds=(-100, 100))
    raw_portfolio_comp = ef.max_quadratic_utility(risk_aversion=portfolio_spec["risk_aversion"])

    # Convert cleaned weights to DataFrame
    portfolio_weights_df = pd.DataFrame(list(ef.clean_weights().items()), columns=['Stock', 'Weight'])
    portfolio_weights_df.set_index('Stock', inplace=True)
    portfolio_weights_df = portfolio_weights_df.drop("RISK_FREE")

    # Check that it is the same as doing 1/gamma * Sigma^{-1}mu
    if CHECK:
        # Calculate the inverse of the covariance matrix
        inv_covariance = np.linalg.inv(covariance_log_returns.values)

        # Calculate portfolio weights for the tangency portfolio
        weights = 1 / portfolio_spec["risk_aversion"] * np.dot(inv_covariance, mean_log_returns.values)

        # Create a DataFrame for the portfolio composition
        portfolio_weights_df_check = pd.DataFrame(weights, index = covariance_log_returns.index, columns = ['Weight'])

        are_equal = np.isclose(portfolio_weights_df_check, portfolio_weights_df, rtol = 1e-4,
                               atol = 1e-4).all().all()

        if not are_equal:
            raise ValueError(f"Shrinkage weights are not consistent.")

    return portfolio_weights_df

def calculate_black_litterman_portfolio(portfolio_spec,
                                        trading_date_ts,
                                        k_stock_market_caps_df,
                                        k_stock_prices_df,
                                        risk_free_rate_df):
    logger.info(f"Calculating Black-Litterman portfolio weights.")
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec,
        trading_date_ts,
        k_stock_prices_df)

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df
    )

    k_stock_market_caps_latest_df = k_stock_market_caps_df.iloc[-1].sort_values(ascending=False)

    window_annualization_factor = get_window_annualization_factor(portfolio_spec)

    # Covariance matrix
    covariance_log_returns = risk_models.CovarianceShrinkage(k_stock_excess_log_returns_window_df,
                                                             returns_data=True,
                                                             frequency = window_annualization_factor).ledoit_wolf()

    viewdict = {}
    market_prior = black_litterman.market_implied_prior_returns(k_stock_market_caps_latest_df.squeeze(),
                                                                portfolio_spec["risk_aversion"],
                                                                covariance_log_returns,
                                                                risk_free_rate = 0)

    bl = BlackLittermanModel(covariance_log_returns, pi=market_prior, absolute_views=viewdict)
    bl_mean_log_returns = bl.bl_returns()
    bl_covariance_log_returns = bl.bl_cov()

    # Add risk free asset
    bl_mean_log_returns_with_risk_free_asset = bl_mean_log_returns.copy()
    bl_mean_log_returns_with_risk_free_asset["RISK_FREE"] = 0

    bl_covariance_log_returns_with_risk_free_asset = bl_covariance_log_returns.copy()
    bl_covariance_log_returns_with_risk_free_asset["RISK_FREE"] = 0
    bl_covariance_log_returns_with_risk_free_asset.loc["RISK_FREE"] = 0

    ef = EfficientFrontier(bl_mean_log_returns_with_risk_free_asset, bl_covariance_log_returns_with_risk_free_asset, weight_bounds=(-100, 100))
    raw_portfolio_comp = ef.max_quadratic_utility(risk_aversion=portfolio_spec["risk_aversion"])

    # Convert cleaned weights to DataFrame
    portfolio_weights_df = pd.DataFrame(list(ef.clean_weights().items()), columns=['Stock', 'Weight'])
    portfolio_weights_df.set_index('Stock', inplace=True)
    portfolio_weights_df = portfolio_weights_df.drop("RISK_FREE")

    return portfolio_weights_df

def calculate_conjugate_hf_mcm_portfolio(portfolio_spec,
                                         trading_date_ts,
                                         k_stock_market_caps_df,
                                         k_stock_prices_df,
                                         k_stock_intraday_prices_df,
                                         mcm_dict,
                                         risk_free_rate_df):
    logger.info(f"Calculating conjugate high frequency MCM portfolio weights.")

    mean_conjugate_posterior_nu_df = calculate_mean_conjugate_posterior_nu(portfolio_spec,
                                                                        trading_date_ts,
                                                                        k_stock_prices_df,
                                                                        k_stock_market_caps_df,
                                                                        k_stock_intraday_prices_df,
                                                                        mcm_dict,
                                                                        risk_free_rate_df)

    return 1 / portfolio_spec["risk_aversion"] * mean_conjugate_posterior_nu_df

def calculate_jeffreys_portfolio(portfolio_spec,
                                 trading_date_ts,
                                 k_stock_prices_df,
                                 risk_free_rate_df):
    logger.info(f"Calculating Jeffreys portfolio weights.")

    mean_jeffreys_posterior_nu_df = calculate_mean_jeffreys_posterior_nu(portfolio_spec,
                                                                        trading_date_ts,
                                                                        k_stock_prices_df,
                                                                        risk_free_rate_df)

    return 1 / portfolio_spec["risk_aversion"] * mean_jeffreys_posterior_nu_df

def calculate_jorion_portfolio(portfolio_spec,
                              trading_date_ts,
                              k_stock_prices_df,
                              risk_free_rate_df):

    logger.info(f"Calculating Jorion portfolio.")

    # Adjust the stock prices DataFrame based on rolling window
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df
    )

    # Using notations of Bayesian Portfolio Analysis (2010) by Avramov and Zhou
    N = len(k_stock_prices_df.columns)
    T = len(k_stock_excess_log_returns_window_df)

    # Sample mean
    mu_hat_df = k_stock_excess_log_returns_window_df.mean().to_frame()

    # Sample covariance
    V_hat_df = k_stock_excess_log_returns_window_df.cov()

    # Shrinkage
    V_bar_df = T / (T - N - 2) * V_hat_df
    V_bar_inverse_df = pd.DataFrame(np.linalg.inv(V_bar_df.to_numpy()), index=V_bar_df.index, columns=V_bar_df.columns)
    one_N_df = pd.DataFrame(np.ones(N), index=V_bar_inverse_df.index)
    mu_hat_g = (one_N_df.T.dot(V_bar_inverse_df).dot(mu_hat_df) / one_N_df.T.dot(V_bar_inverse_df).dot(one_N_df)).values[0,0]

    mu_hat_difference = mu_hat_df.sub(mu_hat_g * one_N_df.values, axis=0)
    lambda_hat = (N + 2) / (mu_hat_difference.T.dot(V_bar_inverse_df).dot(mu_hat_difference)).values[0, 0]

    v_hat = (N + 2) / ((N + 2) + T * mu_hat_difference.T.dot(V_bar_inverse_df).dot(mu_hat_difference)).values[0, 0]
    V_hat_PJ = (1 + 1 / (T + lambda_hat)) * V_bar_df + lambda_hat / (T * (T + 1 + lambda_hat)) * one_N_df.dot(one_N_df.T) / (one_N_df.T.dot(V_bar_inverse_df).dot(one_N_df)).values[0, 0]
    mu_hat_PJ_df = (1 - v_hat) * mu_hat_df + v_hat * mu_hat_g * one_N_df

    V_hat_PJ_inverse_df = pd.DataFrame(np.linalg.inv(V_hat_PJ.to_numpy()), index=V_hat_PJ.index, columns=V_hat_PJ.columns)

    portfolio_weights_df = 1 / portfolio_spec["risk_aversion"] * V_hat_PJ_inverse_df.dot(mu_hat_PJ_df).reset_index().rename(columns={'index': 'Stock', 0: 'Weight'}).set_index('Stock')

    return portfolio_weights_df

def calculate_greyserman_portfolio(portfolio_spec,
                                      trading_date_ts,
                                      k_stock_prices_df,
                                      risk_free_rate_df):

    logger.info(f"Calculating Greyserman portfolio.")

    # Adjust the stock prices DataFrame based on the portfolio's rebalancing frequency and rolling window
    k_stock_prices_window_df = adjust_stock_prices_window(
        portfolio_spec, trading_date_ts, k_stock_prices_df
    )

    # Calculate the excess log returns for the adjusted stock prices
    k_stock_excess_log_returns_window_df = calculate_excess_log_returns_from_prices(
        portfolio_spec, k_stock_prices_window_df, risk_free_rate_df
    )

    n = len(k_stock_excess_log_returns_window_df)
    k = len(k_stock_prices_df.columns)
    x_bar_df = k_stock_excess_log_returns_window_df.mean().to_frame()
    S_df = k_stock_excess_log_returns_window_df.cov()
    S_h_df = pd.DataFrame(np.where(np.eye(k) == 1, 1, 0.5), index=S_df.index[:k], columns=S_df.index[:k])
    one_N_df = pd.DataFrame(np.ones(k), index=x_bar_df.index)
    kappa_h = round(0.1 * n)
    nu_h = k
    weights_b_storage = []
    # Using notations of Incorporating different sources of information for Bayesian optimal portfolio selection by Bodnar et al.
    for i in range(1000):
        xi_b = np.random.uniform(-1000, 1000)
        eta_b = gamma.rvs(a=1, scale=10)
        a_h = 1 / (n + kappa_h) * (n * x_bar_df + kappa_h * xi_b * one_N_df)
        D_h = (n - 1) * S_df + eta_b * S_h_df + n * x_bar_df.dot(x_bar_df.T) + kappa_h * xi_b**2 * one_N_df.dot(one_N_df.T) - (n + kappa_h) * a_h.dot(a_h.T)
        D_h_inverse = pd.DataFrame(np.linalg.inv(D_h.to_numpy()), index=D_h.index, columns=D_h.columns)
        weights_b_df = 1 / portfolio_spec["risk_aversion"] * (nu_h + n + 1) * (1 - 1 / (nu_h + n - k)) * D_h_inverse.dot(a_h)
        weights_b_storage.append(weights_b_df)

    weights_b_all_df = pd.concat(weights_b_storage, axis=1)
    weights_b_df_mean = weights_b_all_df.mean(axis=1)

    weights_b_df_mean.index.name = 'Stock'
    weights_b_df_mean.name = 'Weight'
    return weights_b_df_mean.to_frame(name='Weight')


def calculate_portfolio_weights(trading_date_ts,
                                portfolio_spec,
                                market_data):
    logger.info(f"Calculating portfolio weights for {portfolio_spec['weighting_strategy']} on {trading_date_ts}")

    stock_market_caps_df = market_data["stock_market_caps_df"]
    stock_prices_df = market_data["stock_prices_df"]
    stock_intraday_prices_df = market_data["stock_intraday_prices_df"]
    risk_free_rate_df = market_data["risk_free_rate_df"]

    #  Create a dictionary of available MCMs (Market Condition Metrics)
    mcm_dict = {
        "VIX": market_data.get("vix_prices_df"),
        "EPU": market_data.get("epu_prices_df"),
        "ClimateRisk1": market_data.get("climate_risk1_df"),
        "ClimateRisk2": market_data.get("climate_risk2_df")
    }

    #  Remove any MCMs that are missing (None values)
    mcm_dict = {key: value for key, value in mcm_dict.items() if value is not None}

    #  Get k largest stocks and market caps at trading_date_ts
    k_stock_market_caps_trading_date_df = get_k_largest_stocks_market_caps(
        stock_market_caps_df, stock_prices_df, stock_intraday_prices_df,
        trading_date_ts, portfolio_spec["size"], get_window_trading_days(portfolio_spec),
        portfolio_spec["rebalancing_frequency"])

    #  Filter market caps & stock prices to only include relevant data
    k_stock_market_caps_df = stock_market_caps_df.loc[:trading_date_ts, 
                                                       k_stock_market_caps_trading_date_df.index.intersection(stock_market_caps_df.columns)]
    k_stock_prices_df = stock_prices_df.loc[:trading_date_ts, 
                                             k_stock_market_caps_trading_date_df.index.intersection(stock_prices_df.columns)]

    #  Filter intraday stock prices
    trading_date_ts_inclusive = pd.Timestamp(trading_date_ts).replace(hour=23, minute=59, second=59)
    k_stock_intraday_prices_df = stock_intraday_prices_df.loc[
        stock_intraday_prices_df.index <= trading_date_ts_inclusive, 
        k_stock_market_caps_trading_date_df.index.intersection(stock_intraday_prices_df.columns)
    ]

    #  Ensure MCMs only include data until `trading_date_ts`
    for key in list(mcm_dict.keys()):  # Iterate over a copy of keys to avoid runtime errors
        if trading_date_ts not in mcm_dict[key].index:
            logger.warning(f"{key} data is missing for {trading_date_ts}. Removing from MCM list.")
            del mcm_dict[key]  # Remove MCM if no data for the trading date
        else:
            mcm_dict[key] = mcm_dict[key].loc[mcm_dict[key].index <= trading_date_ts]

    #  Check for NA values in stock prices
    if k_stock_prices_df.tail(get_window_trading_days(portfolio_spec)).isna().any().any():
        logger.error(f"Found NA values in the filtered stock prices.")
        raise ValueError(f"The filtered stock prices contain NA values.")

    #  Portfolio weight calculations
    if portfolio_spec["weighting_strategy"] == "vw":
        portfolio_weights_df = calculate_value_weighted_portfolio(portfolio_spec, trading_date_ts, k_stock_market_caps_df)

    elif portfolio_spec["weighting_strategy"] == "ew":
        portfolio_weights_df = calculate_equally_weighted_portfolio(portfolio_spec, k_stock_prices_df)

    elif portfolio_spec["weighting_strategy"] == "shrinkage":
        portfolio_weights_df = calculate_shrinkage_portfolio(portfolio_spec, trading_date_ts, k_stock_prices_df, risk_free_rate_df)

    elif portfolio_spec["weighting_strategy"] == "black_litterman":
        portfolio_weights_df = calculate_black_litterman_portfolio(portfolio_spec, trading_date_ts, k_stock_market_caps_df, k_stock_prices_df, risk_free_rate_df)

    #  Handle Bayesian Conjugate Models with Multiple Risk Factors
    elif "conjugate_hf" in portfolio_spec["weighting_strategy"]:
        if not mcm_dict:
            logger.error("No MCMs (risk factors) available for conjugate Bayesian portfolio.")
            raise ValueError("No valid MCMs provided for Bayesian portfolio.")

        logger.info(f"Using the following MCMs: {list(mcm_dict.keys())}")

        portfolio_weights_df = calculate_conjugate_hf_mcm_portfolio(
            portfolio_spec,
            trading_date_ts,
            k_stock_market_caps_df,
            k_stock_prices_df,
            k_stock_intraday_prices_df,
            mcm_dict,  #  Pass the MCM dictionary dynamically
            risk_free_rate_df
        )

    elif portfolio_spec["weighting_strategy"] == "jeffreys":
        portfolio_weights_df = calculate_jeffreys_portfolio(portfolio_spec, trading_date_ts, k_stock_prices_df, risk_free_rate_df)

    elif portfolio_spec["weighting_strategy"] == "jorion":
        portfolio_weights_df = calculate_jorion_portfolio(portfolio_spec, trading_date_ts, k_stock_prices_df, risk_free_rate_df)

    elif portfolio_spec["weighting_strategy"] == "greyserman":
        portfolio_weights_df = calculate_greyserman_portfolio(portfolio_spec, trading_date_ts, k_stock_prices_df, risk_free_rate_df)

    else:
        logger.error(f"Unknown weighting strategy: {portfolio_spec['weighting_strategy']}")
        raise ValueError(f"Unknown weighting strategy: {portfolio_spec['weighting_strategy']}")

    return portfolio_weights_df


def compute_portfolio_turnover(portfolio_weights_before_df, portfolio_weights_after_df):

    # Merging the old and new weights with a suffix to differentiate them
    portfolio_weights_before_after_df = portfolio_weights_before_df.merge(portfolio_weights_after_df,
                                                how='outer',
                                                left_index=True,
                                                right_index=True,
                                                suffixes=('_before', '_after'))

    # Fill missing values with 0s (for new stocks or those that have been removed)
    portfolio_weights_before_after_df.fillna(0, inplace=True)

    # Calculate absolute difference for each stock and then compute turnover
    portfolio_weights_before_after_df['weight_diff'] = abs(portfolio_weights_before_after_df['Weight_before'] - portfolio_weights_before_after_df['Weight_after'])

    # Calculate turnover corresponding to risk free asset
    risk_free_turnover = abs(portfolio_weights_before_df['Weight'].sum() - portfolio_weights_after_df['Weight'].sum())

    # Calculate total turnover
    turnover = (portfolio_weights_before_after_df['weight_diff'].sum() + risk_free_turnover) / 2

    return turnover

def calculate_average_distance_to_comparison_portfolio(portfolio_weights_df,
                                                  portfolio_spec,
                                                  trading_date_ts,
                                                  market_data,
                                                  comparison_portfolio_weighting_strategy):
    comparison_portfolio_spec = {"size": portfolio_spec["size"],
                                 "rebalancing_frequency": portfolio_spec["rebalancing_frequency"],
                                 "rolling_window": portfolio_spec["rolling_window"],
                                 "rolling_window_frequency": portfolio_spec["rolling_window_frequency"]}

    if comparison_portfolio_weighting_strategy == "vw":
        comparison_portfolio_spec["weighting_strategy"] = "vw"
    else:
        raise ValueError("Unknown comparison portfolio.")

    comparison_portfolio_weights_df = calculate_portfolio_weights(trading_date_ts,
                                                                    comparison_portfolio_spec,
                                                                    market_data)

    # Ensure portfolio_weights_df and comparison_portfolio_weights_df have exactly the same stocks
    if not portfolio_weights_df.index.equals(comparison_portfolio_weights_df.index):
        raise ValueError("The portfolios do not match exactly in terms of stocks.")

    # Compute average distance between weights using L1 norm
    scaling = portfolio_spec["risk_aversion"] if portfolio_spec.get("risk_aversion") is not None else 1
    average_distance = np.abs(portfolio_weights_df * scaling - comparison_portfolio_weights_df).mean().item()

    return average_distance

class Portfolio:

    def get_portfolio_simple_returns(self):
        return self.portfolio_simple_returns_series

    def get_portfolio_turnover(self):
        return self.portfolio_turnover_series

    def get_portfolio_weights_metrics(self):
        return self.portfolio_weights_metrics_df

    def __init__(self,
                 ts_start_date,
                 portfolio_spec):
        self.ts_start_date = ts_start_date
        self.portfolio_spec = portfolio_spec
        self.portfolio_simple_returns_series = pd.Series(dtype = "float64", name = portfolio_spec["display_name"])
        self.portfolio_turnover_series = pd.Series(dtype = "float64", name = portfolio_spec["display_name"])
        self.portfolio_weights_metrics_df = pd.DataFrame(dtype = "float64")
        self.last_rebalance_date_ts = None

    def update_portfolio(self,
                         trading_date_ts,
                         market_data):

        # Calculate daily portfolio return
        if self.ts_start_date != trading_date_ts:
            # Filter out stocks not in the portfolio
            filtered_stock_simple_returns_series = market_data["stock_simple_returns_df"].loc[trading_date_ts].reindex(self.portfolio_weights_df.index)

            # Multiply returns by weights element-wise and then sum to get the portfolio return
            portfolio_simple_return = (filtered_stock_simple_returns_series * self.portfolio_weights_df['Weight']).sum()

            # Add risk-free return
            risk_free_rate_df = market_data["risk_free_rate_df"]
            most_recent_risk_free_rate = risk_free_rate_df.asof(trading_date_ts).iloc[0]
            risk_free_daily_return = ((most_recent_risk_free_rate + 1) ** (1 / 252) - 1)
            portfolio_simple_return += (1 - self.portfolio_weights_df['Weight'].sum()) * risk_free_daily_return

            self.portfolio_simple_returns_series[trading_date_ts] = portfolio_simple_return

            # Update weight for the risk-free asset
            current_risk_free_weight = 1 - self.portfolio_weights_df['Weight'].sum()
            updated_risk_free_weight = current_risk_free_weight * (1 + risk_free_daily_return)

            # Update weights for the stocks
            self.portfolio_weights_df['Weight'] = (
                        self.portfolio_weights_df['Weight'] * (1 + filtered_stock_simple_returns_series))

            # Update the total invested value by adding the updated risk-free weight
            total_value = self.portfolio_weights_df['Weight'].sum() + updated_risk_free_weight

            # Normalize the weights so they sum up to 1
            self.portfolio_weights_df['Weight'] = self.portfolio_weights_df['Weight'] / total_value

            # Check that weights sum to 1
            if abs((self.portfolio_weights_df['Weight'].values.sum() + updated_risk_free_weight / total_value) - 1) > 1e-5:
                logger.error(f"Weights do not sum to 1.")
                raise ValueError(f"Weights do not sum to 1.")

        if self.last_rebalance_date_ts is None:
            rebalance = True
        elif self.portfolio_spec["rebalancing_frequency"] == "daily":
            rebalance = True
        elif self.portfolio_spec["rebalancing_frequency"] == "weekly":
            rebalance = trading_date_ts.weekday() == 2 or (trading_date_ts - self.last_rebalance_date_ts).days > 7
        elif self.portfolio_spec["rebalancing_frequency"] == "monthly":
            rebalance = trading_date_ts.month != self.last_rebalance_date_ts.month
        else:
            logger.error(f"Unknown rebalancing frequency.")
            raise ValueError(f"Unknown rebalancing frequency.")

        if rebalance:
            if not self.last_rebalance_date_ts is None:
                # Make a copy of the current weights to calculate turnover later
                portfolio_weights_before_df = self.portfolio_weights_df.copy()

            # Calculate the new portfolio weights
            self.portfolio_weights_df = calculate_portfolio_weights(trading_date_ts,
                                                                 self.portfolio_spec,
                                                                 market_data)

            average_distance_to_comparison_portfolio = calculate_average_distance_to_comparison_portfolio(self.portfolio_weights_df,
                                                                                                          self.portfolio_spec,
                                                                                                          trading_date_ts,
                                                                                                          market_data,
                                                                                                          "vw")

            portfolio_weights_max_long = self.portfolio_weights_df ['Weight'][self.portfolio_weights_df['Weight'] > 0].max()
            portfolio_weights_max_short = self.portfolio_weights_df ['Weight'][self.portfolio_weights_df['Weight'] < 0].min()
            portfolio_weights_avg_long = self.portfolio_weights_df ['Weight'][self.portfolio_weights_df['Weight'] > 0].mean()
            portfolio_weights_avg_short = self.portfolio_weights_df ['Weight'][self.portfolio_weights_df['Weight'] < 0].mean()
            portfolio_weights_avg_distance = average_distance_to_comparison_portfolio

            # Prepare a DataFrame row to append
            portfolio_weights_metrics_date_df = pd.DataFrame({
                "max_long": [portfolio_weights_max_long],
                "max_short": [portfolio_weights_max_short],
                "avg_long": [portfolio_weights_avg_long],
                "avg_short": [portfolio_weights_avg_short],
                "average_distance_to_comparison_portfolio": [portfolio_weights_avg_distance]
            }, index = [trading_date_ts])

            self.portfolio_weights_metrics_df = pd.concat([self.portfolio_weights_metrics_df, portfolio_weights_metrics_date_df])

            if not self.last_rebalance_date_ts is None:
                turnover = compute_portfolio_turnover(portfolio_weights_before_df, self.portfolio_weights_df)
                self.portfolio_turnover_series[trading_date_ts] = turnover
                turnover_cost = self.portfolio_spec["turnover_cost"] / 10000 * turnover
                self.portfolio_simple_returns_series[trading_date_ts] -= turnover_cost

            logger.info(f"Portfolio size {trading_date_ts}: {len(self.portfolio_weights_df.index)}")

            self.last_rebalance_date_ts = trading_date_ts

def backtest_portfolio(portfolio_spec,
                       ts_start_date,
                       ts_end_date,
                       market_data):

    # Trading dates
    trading_date_ts = [pd.Timestamp(ts) for ts in market_data["stock_prices_df"].index]
    trading_date_ts = [ts for ts in trading_date_ts if ts_start_date <= ts <= ts_end_date]

    portfolio = Portfolio(trading_date_ts[0], portfolio_spec)

    for trading_date_ts in trading_date_ts:
        portfolio.update_portfolio(trading_date_ts,
                                   market_data)

    return {"portfolio_simple_returns_series": portfolio.get_portfolio_simple_returns(),
            "portfolio_turnover_series": portfolio.get_portfolio_turnover(),
            "portfolio_weights_metrics_df": portfolio.get_portfolio_weights_metrics()}
