import os
import pandas as pd
import quantstats as qs
from dotenv import load_dotenv
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import PercentFormatter
import matplotlib as mpl
import portfolio_specs
from scipy.stats import norm, skew, kurtosis
import numpy as np
import re
import seaborn as sns
import math


load_dotenv()
logging_level = os.environ.get("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

CHECK = True
def get_insolvent_date(returns_series):
    # Calculate cumulative returns
    cumulative_returns = (1 + returns_series).cumprod() - 1

    threshold_date = cumulative_returns[cumulative_returns < -0.99].first_valid_index()

    return threshold_date


def adjust_weights(weights_df, returns_series):
    insolvent_date = get_insolvent_date(returns_series)

    if insolvent_date is not None:
        # Set weights to 0 for all dates after the threshold date
        weights_df.loc[weights_df.index > insolvent_date] = 0

    return weights_df


def adjust_returns(series):
    # Initialize the adjusted series with the original series values
    adjusted_series = series.copy()

    for i in range(len(series)):
        # Calculate cumulative return up to this point
        cum_return = (1 + adjusted_series.iloc[:i + 1]).cumprod().iloc[-1] - 1

        # If cum_return drops below -100%, adjust the last return before this point
        if cum_return < -1:
            if i > 0:
                # Calculate the required return to make cumulative return exactly -100%
                # We use the cumulative return just before the drop below -100% to adjust the return at i
                prev_cum_return = (1 + adjusted_series.iloc[:i]).cumprod().iloc[-1] - 1
                # Adjust the return at i to bring cumulative return exactly to -100%
                adjusted_return = 0.000001 / prev_cum_return - 1
                adjusted_series.iloc[i] = adjusted_return
            else:
                # If the first return in the series leads to a cumulative return below -100%
                # directly adjust it to represent a -100% cumulative return
                adjusted_series.iloc[i] = -1

            # Set all subsequent returns to 0 after adjustment to avoid further decline
            adjusted_series.iloc[i + 1:] = 0
            break

    return adjusted_series

def format_pct_axis(x, pos):
    return f"{x * 100:.0f}%"


def prob_sharpe_ratio_with_benchmark(excess_simple_returns_series,
                                     excess_benchmark_simple_returns_series):
    # Calculate Sharpe ratios for both the strategy and the benchmark
    benchmark_sharpe_ratio = qs.stats.sharpe(excess_benchmark_simple_returns_series, periods=1)
    sharpe_ratio = qs.stats.sharpe(excess_simple_returns_series, periods=1)

    # Manual checks: compare with manual calculations
    if CHECK:
        mean_excess_return_benchmark = np.mean(excess_benchmark_simple_returns_series)
        std_dev_benchmark = np.std(excess_benchmark_simple_returns_series, ddof=1)
        benchmark_sharpe_ratio_manual = mean_excess_return_benchmark / std_dev_benchmark

        mean_excess_return_strategy = np.mean(excess_simple_returns_series)
        std_dev_strategy = np.std(excess_simple_returns_series, ddof=1)
        sharpe_ratio_manual = mean_excess_return_strategy / std_dev_strategy

        tolerance = 1e-5
        assert np.isclose(benchmark_sharpe_ratio, benchmark_sharpe_ratio_manual, atol=tolerance), "Benchmark Sharpe Ratios do not match."
        assert np.isclose(sharpe_ratio, sharpe_ratio_manual, atol=tolerance), "Strategy Sharpe Ratios do not match."

    # Calculate skewness and kurtosis from the raw return series
    skewness = skew(excess_simple_returns_series)
    sample_kurtosis = kurtosis(excess_simple_returns_series, fisher=False)  # Not excess kurtosis

    # Compute manual skewness and kurtosis checks safely
    if CHECK:
        try:
            sample_skewness_check = ((excess_simple_returns_series - np.mean(excess_simple_returns_series)) ** 3).mean() / (np.std(excess_simple_returns_series, ddof=1) ** 3)
        except ZeroDivisionError:
            sample_skewness_check = np.nan
        try:
            sample_kurtosis_check = ((excess_simple_returns_series - np.mean(excess_simple_returns_series)) ** 4).mean() / (np.std(excess_simple_returns_series, ddof=1) ** 4)
        except ZeroDivisionError:
            sample_kurtosis_check = np.nan

        print(f"Computed Skewness: {skewness}, Expected Skewness: {sample_skewness_check}")
        print(f"Computed Kurtosis: {sample_kurtosis}, Expected Kurtosis: {sample_kurtosis_check}")

        assert np.isclose(skewness, sample_skewness_check, rtol=1e-2, atol=1e-2), "Sample skewness check failed"
        assert np.isclose(sample_kurtosis, sample_kurtosis_check, rtol=5e-2, atol=5e-2), "Sample kurtosis check failed"

    n = len(excess_simple_returns_series)

    # Variance of the Sharpe ratio, adjusted for skewness and kurtosis
    sharp_ratio_variance = (1 - skewness * sharpe_ratio + ((sample_kurtosis - 1) / 4) * sharpe_ratio ** 2) / (n - 1)
    sigma_sharpe_ratio = np.sqrt(sharp_ratio_variance)

    # Calculate the Probabilistic Sharpe Ratio
    prob_sharpe_ratio = norm.cdf((sharpe_ratio - benchmark_sharpe_ratio) / sigma_sharpe_ratio)

    return prob_sharpe_ratio


def plot_mcm_vs_sp500(mcm_index_prices_df,
                              sp500_simple_returns_df,
                              mcm_index_name):
    # Calculate the cumulative returns for the market index
    sp500_cumulative_returns_df = (1 + sp500_simple_returns_df).cumprod() - 1

    # Create the figure and axis objects
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Formatting
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)

    # main title
    fig.suptitle(f"{mcm_index_name} Values and S&P 500 Cumulative Returns", y=0.96, fontweight="bold", fontsize=14, color="black")

    # Date range
    date_range = "{} - {}".format(
        sp500_cumulative_returns_df.index[0].strftime('%d %b \'%y'),
        sp500_cumulative_returns_df.index[-1].strftime('%d %b \'%y')
    )

    fig.text(0.5, 0.91, date_range, ha='center', va='center', fontsize=12, color="gray")

    ax1.set_facecolor("white")
    fig.set_facecolor("white")

    # Plotting the VIX Prices
    ax1.plot(mcm_index_prices_df.index, mcm_index_prices_df, label=f"{mcm_index_name} Values", color=(0.5, 0.5, 0.5))
    ax1.set_ylabel(f"{mcm_index_name} Values", color=(0.5, 0.5, 0.5))
    ax1.tick_params(axis='y', labelcolor=(0.5, 0.5, 0.5))

    # Create another y-axis for the S&P 500 Cumulative Returns
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(False)
    ax2.plot(sp500_cumulative_returns_df.index, sp500_cumulative_returns_df, label="S&P 500 Cumulative Returns",
             color=(79 / 255, 139 / 255, 188 / 255))
    ax2.set_ylabel("S&P 500 Cumulative Returns", color=(79 / 255, 139 / 255, 188 / 255))
    ax2.tick_params(axis='y', labelcolor=(79 / 255, 139 / 255, 188 / 255))
    ax2.yaxis.set_major_formatter(FuncFormatter(format_pct_axis))

    # Legend below the figure
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=11, ncol=2)

    # Show grid
    ax1.grid(True, linestyle='-', linewidth=0.5, color='gray')

    # Rotate and align the tick labels
    fig.autofmt_xdate()
    ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    # Save the figure
    plt.savefig(f"../results2/{mcm_index_name}_vs_sp500.pdf", dpi=400, bbox_inches='tight')

def plot_cagr_vs_trading_cost(portfolio_specs_simple_returns,
                              risk_aversion,
                              evaluation_period_str):
    # Initialize a dictionary to hold Sharpe ratios for each portfolio display name
    cagr_dfs = {}

    for portfolio_spec, portfolio_simple_returns_series in portfolio_specs_simple_returns.items():
        if f'risk_aversion_{risk_aversion}' in portfolio_spec or f'risk_aversion_NA' in portfolio_spec:
            display_name = portfolio_specs.get_display_name_from_full_name(portfolio_spec)
            # Check insolvent
            insolvent_date = get_insolvent_date(portfolio_simple_returns_series)
            if insolvent_date is None:
                # Calculate CAGR
                portfolio_cagr = qs.stats.cagr(portfolio_simple_returns_series, periods=365)
                match = re.search(r"turnover_cost_(\d+)", portfolio_spec)
                number = int(match.group(1)) if match else None

                # Initialize the dictionary for this display_name if it does not exist
                if display_name not in cagr_dfs:
                    cagr_dfs[display_name] = {}

                # Store the Sharpe ratio
                cagr_dfs[display_name][number] = portfolio_cagr

    # Convert the dictionaries to a DataFrame
    df = pd.DataFrame(cagr_dfs)

    fontname = "Arial"
    title_fontsize = 1.2 * 14
    subtitle_fontsize = 1.2 * 12
    axis_label_fontsize = 1.2 * 12
    tick_label_fontsize = 1.2 * 11
    legend_fontsize = 1.2 * 11

    # Adjusting plot style with seaborn
    sns.set_theme(style = "whitegrid")

    # Create figure with the specified size and resolution
    fig, ax = plt.subplots(figsize = (9.8, 7), dpi = 400)

    # Plot each series with the specified line width
    lw = 1.5
    for i, column in enumerate(df.columns):
        ax.plot(df.index, df[column], marker = 'o', lw = lw, label = column, color = portfolio_specs.get_color_from_display_name(column))

    # Setting title with font adjustments
    ax.set_title('CAGR vs Transaction Cost', fontsize = title_fontsize, fontweight = 'bold',
                 fontname = fontname, color = "black")

    # Setting axis labels with font adjustments
    ax.set_xlabel('Transaction Cost (bps)', fontsize = axis_label_fontsize, fontname = fontname, fontweight = "bold")
    ax.set_ylabel('CAGR', fontsize=axis_label_fontsize, fontname=fontname, fontweight="bold")

    # Formatting y-axis as percentage
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # Adjusting tick labels font
    ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

    # Adjusting legend with the modified position below the plot, in three columns
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.2), fontsize = legend_fontsize, ncol = 3)

    # Apply layout adjustments similar to plot_rolling_stats
    plt.tight_layout()

    # Save the plot as a PDF with tight bounding box
    filename = f"../results2/cagr_vs_trading_cost_risk_aversion_{risk_aversion}_{evaluation_period_str}.pdf"
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()


def plot_sharpe_ratio_vs_trading_cost(portfolio_specs_excess_simple_returns,
                              risk_aversion,
                                      evaluation_period_str):
    # Initialize a dictionary to hold Sharpe ratios for each portfolio display name
    sharpe_ratios_dfs = {}

    for portfolio_spec, portfolio_excess_simple_returns_series in portfolio_specs_excess_simple_returns.items():
        # Proceed only if a display name was returned
        if f'risk_aversion_{risk_aversion}' in portfolio_spec or f'risk_aversion_NA' in portfolio_spec:
            display_name = portfolio_specs.get_display_name_from_full_name(portfolio_spec)
            # Check insolvent
            insolvent_date = get_insolvent_date(portfolio_excess_simple_returns_series)
            if insolvent_date is None:
                portfolio_sharpe_ratio = qs.stats.sharpe(portfolio_excess_simple_returns_series)
                match = re.search(r"turnover_cost_(\d+)", portfolio_spec)
                number = int(match.group(1)) if match else None

                # Initialize the dictionary for this display_name if it does not exist
                if display_name not in sharpe_ratios_dfs:
                    sharpe_ratios_dfs[display_name] = {}

                # Store the Sharpe ratio
                sharpe_ratios_dfs[display_name][number] = portfolio_sharpe_ratio

    # Convert the dictionaries to a DataFrame
    df = pd.DataFrame(sharpe_ratios_dfs)

    fontname = "Arial"
    title_fontsize = 1.2 * 14
    subtitle_fontsize = 1.2 * 12
    axis_label_fontsize = 1.2 * 12
    tick_label_fontsize = 1.2 * 11
    legend_fontsize = 1.2 * 11

    # Adjusting plot style with seaborn
    sns.set_theme(style = "whitegrid")

    # Create figure with the specified size and resolution
    fig, ax = plt.subplots(figsize = (9.8, 7), dpi = 400)

    # Plot each series with the specified line width
    lw = 1.5
    for i, column in enumerate(df.columns):
        ax.plot(df.index, df[column], marker = 'o', lw = lw, label = column, color = portfolio_specs.get_color_from_display_name(column))

    # Setting title with font adjustments
    ax.set_title('Sharpe Ratio vs Transaction Cost', fontsize = title_fontsize, fontweight = 'bold',
                 fontname = fontname, color = "black")

    # Setting axis labels with font adjustments
    ax.set_xlabel('Transaction Cost (bps)', fontsize = axis_label_fontsize, fontname = fontname, fontweight = "bold")
    ax.set_ylabel('Sharpe Ratio', fontsize = axis_label_fontsize, fontname = fontname, fontweight = "bold")

    # Adjusting tick labels font
    ax.tick_params(axis = 'both', which = 'major', labelsize = tick_label_fontsize)

    # Adjusting legend with the modified position below the plot, in three columns
    ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.2), fontsize = legend_fontsize, ncol = 3)

    # Apply layout adjustments similar to plot_rolling_stats
    plt.tight_layout()

    # Save the plot as a PDF with tight bounding box
    filename = f"../results2/sharpe_ratio_vs_trading_cost_risk_aversion_{risk_aversion}_{evaluation_period_str}.pdf"
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()

def plot_performance(portfolio_specs_simple_returns,
                     portfolio_specs_excess_simple_returns,
                     portfolio_specs_portfolio_weights_metrics,
                     risk_aversion,
                     turnover_cost,
                     evaluation_period_str):
    # Filter columns
    # Create empty DataFrames for storing the filtered columns
    filtered_simple_returns_df = pd.DataFrame()
    filtered_excess_simple_returns_df = pd.DataFrame()
    filtered_turnovers_df = pd.DataFrame()
    filtered_max_long_positions_df = pd.DataFrame()
    filtered_max_short_positions_df = pd.DataFrame()
    filtered_avg_long_positions_df = pd.DataFrame()
    filtered_avg_short_positions_df = pd.DataFrame()
    
    # Create DataFrames for average weight distances for conjugate strategies
    filtered_conjugate_hf_vix_vw_avg_distances_df = pd.DataFrame()
    filtered_conjugate_hf_epu_vw_avg_distances_df = pd.DataFrame()
    filtered_conjugate_hf_climate_risk1_avg_distances_df = pd.DataFrame()
    filtered_conjugate_hf_climate_risk2_avg_distances_df = pd.DataFrame()

    # Loop through each portfolio spec to filter simple and excess returns
    for portfolio_spec, portfolio_simple_returns_series in portfolio_specs_simple_returns.items():
        if (f'risk_aversion_{risk_aversion}' in portfolio_spec or f'risk_aversion_NA' in portfolio_spec) \
           and f'turnover_cost_{turnover_cost}' in portfolio_spec:
            filtered_simple_returns_df[portfolio_simple_returns_series.name] = adjust_returns(portfolio_simple_returns_series)

    for portfolio_spec, portfolio_excess_simple_returns_series in portfolio_specs_excess_simple_returns.items():
        if (f'risk_aversion_{risk_aversion}' in portfolio_spec or f'risk_aversion_NA' in portfolio_spec) \
           and f'turnover_cost_{turnover_cost}' in portfolio_spec:
            filtered_excess_simple_returns_df[portfolio_excess_simple_returns_series.name] = adjust_returns(portfolio_excess_simple_returns_series)
    
    # Filter portfolio weight metrics for max/avg positions
    for portfolio_spec, portfolio_weights_metrics_df in portfolio_specs_portfolio_weights_metrics.items():
        if (f'risk_aversion_{risk_aversion}' in portfolio_spec or f'risk_aversion_NA' in portfolio_spec) \
           and f'turnover_cost_{turnover_cost}' in portfolio_spec:
            display_name = portfolio_specs.get_display_name_from_full_name(portfolio_spec)
            filtered_max_long_positions_df[display_name] = portfolio_weights_metrics_df["max_long"].fillna(0)
            filtered_max_short_positions_df[display_name] = portfolio_weights_metrics_df["max_short"].fillna(0)
            filtered_avg_long_positions_df[display_name] = portfolio_weights_metrics_df["avg_long"].fillna(0)
            filtered_avg_short_positions_df[display_name] = portfolio_weights_metrics_df["avg_short"].fillna(0)
    
    # Filter average weight distances for conjugate high-frequency strategies
    for portfolio_spec, portfolio_weights_metrics_df in portfolio_specs_portfolio_weights_metrics.items():
        if "conjugate" in portfolio_spec and f'risk_aversion_{risk_aversion}' in portfolio_spec \
           and f'turnover_cost_{turnover_cost}' in portfolio_spec:
            match = re.search(r"mcm_scaling_([-+]?\d*\.?\d+)", portfolio_spec)
            if match:
                number = float(match.group(1))
                number_str = f"{number:.0f}" if number.is_integer() else f"{number}"
                if "vix" in portfolio_spec:
                    filtered_conjugate_hf_vix_vw_avg_distances_df[f"MCM = {number_str}\u00D7VIX"] = portfolio_weights_metrics_df["average_distance_to_comparison_portfolio"]
                elif "epu" in portfolio_spec:
                    filtered_conjugate_hf_epu_vw_avg_distances_df[f"MCM = {number_str}\u00D7EPU"] = portfolio_weights_metrics_df["average_distance_to_comparison_portfolio"]
                elif "climate_risk1" in portfolio_spec:
                    filtered_conjugate_hf_climate_risk1_avg_distances_df[f"MCM = {number_str}\u00D7CR1"] = portfolio_weights_metrics_df["average_distance_to_comparison_portfolio"]
                elif "climate_risk2" in portfolio_spec:
                    filtered_conjugate_hf_climate_risk2_avg_distances_df[f"MCM = {number_str}\u00D7CR2"] = portfolio_weights_metrics_df["average_distance_to_comparison_portfolio"]

    print("🔎 Debugging Portfolio Returns DataFrame:")
    print(f"Shape: {filtered_simple_returns_df.shape}")
    print(f"Columns (Portfolio Names): {filtered_simple_returns_df.columns.tolist()}")
    print(f"Index Range: {filtered_simple_returns_df.index.min()} to {filtered_simple_returns_df.index.max()}")


    # Plot daily returns
    qs.plots.returns(filtered_simple_returns_df,
                     savefig=f"../results2/returns_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.pdf")

    # Plot yearly returns
    qs.plots.yearly_returns(filtered_simple_returns_df,
                     savefig=f"../results2/yearly_returns_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.pdf")

    # Plot rolling Sharpe and Sortino ratios, rolling volatility, and drawdowns
    qs.plots.rolling_sharpe(filtered_excess_simple_returns_df,
                     savefig=f"../results2/rolling_sharpe_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.pdf")
    qs.plots.rolling_sortino(filtered_excess_simple_returns_df,
                     savefig=f"../results2/rolling_sortino_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.pdf")
    qs.plots.rolling_volatility(filtered_simple_returns_df,
                     savefig=f"../results2/rolling_volatility_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.pdf")
    qs.plots.drawdown(filtered_simple_returns_df,
                      savefig=f"../results2/drawdown_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.pdf")

    # Plot average weight distances for conjugate strategies if available
    if len(filtered_conjugate_hf_vix_vw_avg_distances_df.columns) > 1:
        qs.plots.weight_distances(filtered_conjugate_hf_vix_vw_avg_distances_df,
                                  comparison_portfolio_type="VW",
                                  ncols=4,
                                  savefig=f"../results2/average_weight_distance_conjugate_hf_vix_vw_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.pdf")
    if len(filtered_conjugate_hf_epu_vw_avg_distances_df.columns) > 1:
        qs.plots.weight_distances(filtered_conjugate_hf_epu_vw_avg_distances_df,
                                  comparison_portfolio_type="VW",
                                  ncols=4,
                                  savefig=f"../results2/average_weight_distance_conjugate_hf_epu_vw_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.pdf")
    if len(filtered_conjugate_hf_climate_risk1_avg_distances_df.columns) > 1:
        qs.plots.weight_distances(filtered_conjugate_hf_climate_risk1_avg_distances_df,
                                  comparison_portfolio_type="VW",
                                  ncols=4,
                                  savefig=f"../results2/average_weight_distance_conjugate_hf_climate_risk1_vw_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.pdf")
    if len(filtered_conjugate_hf_climate_risk2_avg_distances_df.columns) > 1:
        qs.plots.weight_distances(filtered_conjugate_hf_climate_risk2_avg_distances_df,
                                  comparison_portfolio_type="VW",
                                  ncols=4,
                                  savefig=f"../results2/average_weight_distance_conjugate_hf_climate_risk2_vw_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.pdf")



def process_and_highlight_values(metrics_df):
    higher_is_better = {
        'Cum. Return', 'CAGR', 'Sharpe', 'Prob. Sharpe',
        'Sortino', 'Calmar', 'Max. DD',
        'Avg. Loss', 'Avg. Return', 'Avg. Win', 'Best Day',
        'Worst Day', 'Daily VaR', "Avg. Max. Short Position", "Avg. Short Position"
    }

    lower_is_better = {
        'Ann. Vol.', 'Avg. Turnover', "Avg. Max. Long Position", "Avg. Long Position"
    }

    not_percentage = {'Sharpe', 'Sortino', 'Calmar'}

    for row_label, row_series in metrics_df.iterrows():
        if row_label not in higher_is_better and row_label not in lower_is_better:
            raise ValueError(f"Unexpected row label: {row_label}")

        # Handling None values by setting them to -inf or inf based on the metric category
        processed_values = []
        for val in row_series:
            if val is None:
                if row_label in higher_is_better:
                    processed_values.append(-np.inf)  # Set to negative infinity for higher is better
                else:
                    processed_values.append(np.inf)   # Set to infinity for lower is better
            else:
                processed_values.append(val)

        # Convert, round the values and replace -inf/inf back to None for display purposes
        processed_and_converted_values = [
            round(100 * val, 3) if row_label not in not_percentage and val not in [-np.inf, np.inf] else round(val, 3)
            for val in processed_values
        ]

        max_value = max(processed_and_converted_values)
        min_value = min(processed_and_converted_values)

        new_values = []
        for proc_val in processed_and_converted_values:
            if proc_val in [-np.inf, np.inf]:  # Check for -inf/inf values to label them as Worst directly
                str_val = "None (Worst)"
            else:
                str_val = f"{proc_val:.3f}%" if row_label not in not_percentage else f"{proc_val:.3f}"
                if proc_val == max_value:
                    str_val += " (Best)" if row_label in higher_is_better else " (Worst)"
                elif proc_val == min_value:
                    str_val += " (Worst)" if row_label in higher_is_better else " (Best)"

            new_values.append(str_val)

        metrics_df.loc[row_label] = new_values

    return metrics_df


def performance_metrics(portfolio_specs_simple_returns,
                         portfolio_specs_excess_simple_returns,
                        portfolio_specs_turnover,
                        portfolio_specs_portfolio_weights_metrics,
                         risk_aversion,
                         turnover_cost,
                        evaluation_period_str):
    # Filter columns
    # Create empty DataFrames for storing the filtered columns
    filtered_simple_returns_df = pd.DataFrame()
    filtered_excess_simple_returns_df = pd.DataFrame()
    filtered_turnovers_df = pd.DataFrame()
    filtered_max_long_positions_df = pd.DataFrame()
    filtered_max_short_positions_df = pd.DataFrame()
    filtered_avg_long_positions_df = pd.DataFrame()
    filtered_avg_short_positions_df = pd.DataFrame()

    # Loop through each column name to check the conditions
    for portfolio_spec, portfolio_simple_returns_series in portfolio_specs_simple_returns.items():
        if (f'risk_aversion_{risk_aversion}' in portfolio_spec or f'risk_aversion_NA' in portfolio_spec) and f'turnover_cost_{turnover_cost}' in portfolio_spec:
                filtered_simple_returns_df[portfolio_simple_returns_series.name] = adjust_returns(portfolio_simple_returns_series)

    for portfolio_spec, portfolio_excess_simple_returns_series in portfolio_specs_excess_simple_returns.items():
        if (f'risk_aversion_{risk_aversion}' in portfolio_spec or f'risk_aversion_NA' in portfolio_spec) and f'turnover_cost_{turnover_cost}' in portfolio_spec:
                filtered_excess_simple_returns_df[portfolio_excess_simple_returns_series.name] = adjust_returns(portfolio_excess_simple_returns_series)

    for portfolio_spec, portfolio_turnover_series in portfolio_specs_turnover.items():
        if (f'risk_aversion_{risk_aversion}' in portfolio_spec or f'risk_aversion_NA' in portfolio_spec) and f'turnover_cost_{turnover_cost}' in portfolio_spec:
                filtered_turnovers_df[portfolio_turnover_series.name] = portfolio_turnover_series

    for portfolio_spec, portfolio_weights_metrics_df in portfolio_specs_portfolio_weights_metrics.items():
        if (f'risk_aversion_{risk_aversion}' in portfolio_spec or f'risk_aversion_NA' in portfolio_spec) and f'turnover_cost_{turnover_cost}' in portfolio_spec:
            filtered_max_long_positions_df[portfolio_specs.get_display_name_from_full_name(portfolio_spec)] = portfolio_weights_metrics_df["max_long"].fillna(0)
            filtered_max_short_positions_df[portfolio_specs.get_display_name_from_full_name(portfolio_spec)] = portfolio_weights_metrics_df["max_short"].fillna(0)
            filtered_avg_long_positions_df[portfolio_specs.get_display_name_from_full_name(portfolio_spec)] = portfolio_weights_metrics_df["avg_long"].fillna(0)
            filtered_avg_short_positions_df[portfolio_specs.get_display_name_from_full_name(portfolio_spec)] = portfolio_weights_metrics_df["avg_short"].fillna(0)

    # Get insolvent date
    insolvent_dates = {}
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]
        insolvent_date = get_insolvent_date(portfolio_simple_returns_series)
        insolvent_dates[column_name] = insolvent_date

    # Df to store portfolio metrics
    portfolio_specs_metrics_df = pd.DataFrame(columns=filtered_excess_simple_returns_df.columns)

    # Calculate the cumulative return for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]
        portfolio_comp_return = qs.stats.comp(portfolio_simple_returns_series)
        portfolio_specs_metrics_df.at['Cum. Return', column_name] = portfolio_comp_return

    # Calculate the CAGR for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]
        if insolvent_dates[column_name] is None:
            portfolio_cagr = qs.stats.cagr(portfolio_simple_returns_series, periods=365)

            if CHECK:
                portfolio_cagr_check = ((1 + portfolio_simple_returns_series).prod())**(1 / ((portfolio_simple_returns_series.index[-1] - portfolio_simple_returns_series.index[0]).days / 365)) - 1

                is_close = np.isclose(portfolio_cagr, portfolio_cagr_check, atol = 1e-3)
                if not is_close:
                    raise ValueError(f"CAGR is not consistent.")

            portfolio_specs_metrics_df.at['CAGR', column_name] = portfolio_cagr
        else:
            portfolio_specs_metrics_df.at['CAGR', column_name] = None

    # Calculate the Sharpe ratio for each portfolio and add it as a row
    for column_name in filtered_excess_simple_returns_df.columns:
        portfolio_excess_simple_returns_series = filtered_excess_simple_returns_df[column_name]
        if insolvent_dates[column_name] is None:
            portfolio_sharpe_ratio = qs.stats.sharpe(portfolio_excess_simple_returns_series)

            if CHECK:
                portfolio_sharpe_ratio_check = (portfolio_excess_simple_returns_series.mean()) / portfolio_excess_simple_returns_series.std() * (252**0.5)

                is_close = np.isclose(portfolio_sharpe_ratio, portfolio_sharpe_ratio_check, atol = 1e-3)
                if not is_close:
                    raise ValueError(f"Sharpe Ratio is not consistent.")

            portfolio_specs_metrics_df.at['Sharpe', column_name] = portfolio_sharpe_ratio
        else:
            portfolio_specs_metrics_df.at['Sharpe', column_name] = None

    # Calculate the Probabilistic Sharpe ratio for each portfolio and add it as a row
    for column_name in filtered_excess_simple_returns_df.columns:
        portfolio_excess_simple_returns_series = filtered_excess_simple_returns_df[column_name]
        if insolvent_dates[column_name] is None:
            portfolio_prob_sharpe_ratio = prob_sharpe_ratio_with_benchmark(portfolio_excess_simple_returns_series, portfolio_specs_excess_simple_returns['S&P 500'])
            portfolio_specs_metrics_df.at['Prob. Sharpe', column_name] = portfolio_prob_sharpe_ratio
        else:
            portfolio_specs_metrics_df.at['Prob. Sharpe', column_name] = None

    # Calculate the Sortino ratio for each portfolio and add it as a row
    for column_name in filtered_excess_simple_returns_df.columns:
        portfolio_excess_simple_returns_series = filtered_excess_simple_returns_df[column_name]
        if insolvent_dates[column_name] is None:
            portfolio_sortino_ratio = qs.stats.sortino(portfolio_excess_simple_returns_series)
            portfolio_specs_metrics_df.at['Sortino', column_name] = portfolio_sortino_ratio
        else:
            portfolio_specs_metrics_df.at['Sortino', column_name] = None

    # Calculate the Calmar ratio for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]
        if insolvent_dates[column_name] is None:
            portfolio_calmar_ratio = qs.stats.cagr(portfolio_simple_returns_series, periods=365) / abs(qs.stats.max_drawdown(portfolio_simple_returns_series))
            portfolio_specs_metrics_df.at['Calmar', column_name] = portfolio_calmar_ratio
        else:
            portfolio_specs_metrics_df.at['Calmar', column_name] = None

    # Calculate the max drawdown for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_max_drawdown = qs.stats.max_drawdown(portfolio_simple_returns_series)
        portfolio_specs_metrics_df.at['Max. DD', column_name] = portfolio_max_drawdown

    # Calculate the avg loss for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_avg_loss = qs.stats.avg_loss(portfolio_simple_returns_series)

        if CHECK:
            portfolio_avg_loss_check = portfolio_simple_returns_series[portfolio_simple_returns_series < -1e-7].mean()
            is_close = np.isclose(portfolio_avg_loss, portfolio_avg_loss_check, atol=1e-3)
            if not is_close:
                raise ValueError(f"Avg. loss is not consistent.")

        portfolio_specs_metrics_df.at['Avg. Loss', column_name] = portfolio_avg_loss

    # Calculate the avg return for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        if insolvent_dates[column_name] is None:
            portfolio_avg_return = qs.stats.avg_return(portfolio_simple_returns_series)

            if CHECK:
                portfolio_avg_return_check = portfolio_simple_returns_series.mean()
                is_close = np.isclose(portfolio_avg_return, portfolio_avg_return_check, atol = 1e-3)
                if not is_close:
                    raise ValueError(f"Avg. return is not consistent.")
        else:
            portfolio_avg_return = portfolio_simple_returns_series[abs(portfolio_simple_returns_series) > 1e-7].mean()

        portfolio_specs_metrics_df.at['Avg. Return', column_name] = portfolio_avg_return

    # Calculate the avg win for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_avg_win = qs.stats.avg_win(portfolio_simple_returns_series)

        if CHECK:
            portfolio_avg_win_check = portfolio_simple_returns_series[portfolio_simple_returns_series > 1e-7].mean()
            is_close = np.isclose(portfolio_avg_win, portfolio_avg_win_check, atol=1e-3)
            if not is_close:
                raise ValueError(f"Avg. win is not consistent.")

        portfolio_specs_metrics_df.at['Avg. Win', column_name] = portfolio_avg_win

    # Calculate the best day for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        portfolio_best = qs.stats.best(portfolio_simple_returns_series)
        portfolio_specs_metrics_df.at['Best Day', column_name] = portfolio_best

    # Calculate the worst day for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        if insolvent_dates[column_name] is None:
            portfolio_worst = qs.stats.worst(portfolio_simple_returns_series)
        else:
            portfolio_worst = qs.stats.worst(portfolio_simple_returns_series[:insolvent_dates[column_name] - pd.Timedelta(days=1)])
        portfolio_specs_metrics_df.at['Worst Day', column_name] = portfolio_worst

    # Calculate the volatility for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]

        if insolvent_dates[column_name] is None:
            portfolio_volatility = qs.stats.volatility(portfolio_simple_returns_series)

            if CHECK:
                portfolio_volatility_check = portfolio_simple_returns_series.std()*252**0.5
                is_close = np.isclose(portfolio_volatility, portfolio_volatility_check, atol = 1e-3)
                if not is_close:
                    raise ValueError(f"Volatility is not consistent.")
        else:
            portfolio_volatility = portfolio_simple_returns_series[:insolvent_dates[column_name] - pd.Timedelta(days=1)].std()*252**0.5

        portfolio_specs_metrics_df.at['Ann. Vol.', column_name] = portfolio_volatility

    # Calculate the VaR for each portfolio and add it as a row
    for column_name in filtered_simple_returns_df.columns:
        portfolio_simple_returns_series = filtered_simple_returns_df[column_name]
        if insolvent_dates[column_name] is None:
            portfolio_VaR = qs.stats.value_at_risk(portfolio_simple_returns_series)
        else:
            portfolio_VaR = qs.stats.value_at_risk(portfolio_simple_returns_series[:insolvent_dates[column_name]- pd.Timedelta(days=1)])

        portfolio_specs_metrics_df.at['Daily VaR', column_name] = portfolio_VaR

    # for column_name in filtered_max_long_positions_df.columns:
    #     portfolio_max_long_positions_series = filtered_max_long_positions_df[column_name]
    #     portfolio_specs_metrics_df.at['Avg. Max. Long Position', column_name] = portfolio_max_long_positions_series.fillna(0).mean()
    #
    # for column_name in filtered_max_short_positions_df.columns:
    #     portfolio_max_short_positions_series = filtered_max_short_positions_df[column_name]
    #     portfolio_specs_metrics_df.at['Avg. Max. Short Position', column_name] = portfolio_max_short_positions_series.fillna(0).mean()
    #
    # for column_name in filtered_avg_long_positions_df.columns:
    #     portfolio_avg_long_positions_series = filtered_avg_long_positions_df[column_name]
    #     portfolio_specs_metrics_df.at['Avg. Long Position', column_name] = portfolio_avg_long_positions_series.fillna(0).mean()
    #
    # for column_name in filtered_avg_short_positions_df.columns:
    #     portfolio_avg_short_positions_series = filtered_avg_short_positions_df[column_name]
    #     portfolio_specs_metrics_df.at['Avg. Short Position', column_name] = portfolio_avg_short_positions_series.fillna(0).mean()

    # Calculate the turnover for each portfolio and add it as a row
    for column_name in filtered_turnovers_df.columns:
        portfolio_turnover_series = filtered_turnovers_df[column_name]
        if insolvent_dates[column_name] is None:
            portfolio_turnover = portfolio_turnover_series.mean()
        else:
            portfolio_turnover = portfolio_turnover_series[:insolvent_dates[column_name]].mean()

        portfolio_specs_metrics_df.at['Avg. Turnover', column_name] = portfolio_turnover

    highlighted_metrics_df = process_and_highlight_values(portfolio_specs_metrics_df)
    highlighted_metrics_df.to_csv(f"../results2/metrics_risk_aversion_{risk_aversion}_turnover_cost_{turnover_cost}_{evaluation_period_str}.csv", index=True)

def compute_excess_returns(portfolio_simple_returns_series,
                           risk_free_rate_df):
    # Reindex risk_free_rate_df to match the dates of portfolio_simple_returns_series, then forward-fill
    matching_risk_free_rates_series = risk_free_rate_df['DTB3'].reindex(portfolio_simple_returns_series.index).ffill()

    # For any NaNs at the beginning, use backfill to fill them with the next valid value
    matching_risk_free_rates_series = matching_risk_free_rates_series.bfill()

    # Compute the excess returns
    portfolio_excess_simple_returns_series = portfolio_simple_returns_series - (
            (matching_risk_free_rates_series + 1) ** (1 / 252) - 1)

    # Rename the series to match the original portfolio name
    portfolio_excess_simple_returns_series.name = portfolio_simple_returns_series.name

    return portfolio_excess_simple_returns_series


def check_indexes_and_convert_to_datetime(dict_of_dfs):
    indexes_are_equal = True
    common_index = None

    for _, df in dict_of_dfs.items():
        df.index = pd.to_datetime(df.index)  # Convert index to datetime if not already

        if common_index is None:
            common_index = df.index
        else:
            if not common_index.equals(df.index):
                indexes_are_equal = False
                common_index = None  # If indexes do not match, set common_index to None
                break

    return indexes_are_equal, common_index


# Adjust with additional risks
def full_evaluation(portfolio_specs_simple_returns,
                    portfolio_specs_turnover,
                    portfolio_specs_portfolio_weights_metrics,
                    sp500_simple_returns_df,
                    risk_free_rate_df,
                    vix_prices_df,
                    epu_prices_df,
                    climate_risk1_df,
                    climate_risk2_df,
                    evaluation_period_str):

    indexes_are_equal, common_index = check_indexes_and_convert_to_datetime(portfolio_specs_simple_returns)
    filtered_sp500_simple_returns_df = sp500_simple_returns_df.loc[common_index]
    portfolio_specs_simple_returns['S&P 500'] = filtered_sp500_simple_returns_df['S&P 500']
    filtered_vix_prices_df = vix_prices_df.loc[common_index]
    filtered_epu_prices_df = epu_prices_df.loc[common_index]
    filtered_climate_risk1_df = climate_risk1_df.loc[common_index]
    filtered_climate_risk2_df = climate_risk2_df.loc[common_index]

    # Get excess returns
    portfolio_specs_excess_simple_returns = {}
    for portfolio_spec, portfolio_simple_returns_series in portfolio_specs_simple_returns.items():
        portfolio_excess_simple_returns_series = compute_excess_returns(portfolio_simple_returns_series,
                                                                        risk_free_rate_df)

        # Add portfolio_excess_simple_returns_series to the new DataFrame
        portfolio_specs_excess_simple_returns[portfolio_spec] = portfolio_excess_simple_returns_series

    # Performance metrics
    for trading_cost in [15]: #[0, 5, 10, 15, 20, 25, 30, 35]
        for risk_aversion in [5, 10]:
            performance_metrics(portfolio_specs_simple_returns,
                             portfolio_specs_excess_simple_returns,
                            portfolio_specs_turnover,
                            portfolio_specs_portfolio_weights_metrics,
                            risk_aversion,
                             trading_cost,
                             evaluation_period_str)

            plot_performance(portfolio_specs_simple_returns,
                             portfolio_specs_excess_simple_returns,
                             portfolio_specs_portfolio_weights_metrics,
                             risk_aversion,
                             trading_cost,
                             evaluation_period_str)

    # Trading cost vs Sharpe
    for risk_aversion in [5, 10]:
        plot_sharpe_ratio_vs_trading_cost(portfolio_specs_excess_simple_returns,
                                          risk_aversion,
                                          evaluation_period_str)

    # Trading cost vs CAGR
    for risk_aversion in [5, 10]:
        plot_cagr_vs_trading_cost(portfolio_specs_simple_returns,
                                    risk_aversion,
                                   evaluation_period_str)

    # S&P 500 vs VIX
    plot_mcm_vs_sp500(filtered_vix_prices_df, filtered_sp500_simple_returns_df, "VIX")

    # S&P 500 vs EPU
    plot_mcm_vs_sp500(filtered_epu_prices_df, filtered_sp500_simple_returns_df, "EPU")

    # S&P 500 vs Cr1
    plot_mcm_vs_sp500(filtered_climate_risk1_df, filtered_sp500_simple_returns_df, "CR1")

    # S&P 500 vs Cr2
    plot_mcm_vs_sp500(filtered_climate_risk2_df, filtered_sp500_simple_returns_df, "CR2")
