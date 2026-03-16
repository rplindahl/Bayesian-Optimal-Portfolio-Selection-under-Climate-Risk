from itertools import product


def get_color_from_display_name(display_name):
    # Quantstats ['#FFD700', '#E63946', '#A8DADC', '#457B9D', '#FF69B4', '#1D3557', '#F4A261', '#2A9D8F', '#9370DB', '#9DC209']
    colors = {
    "S&P 500": "#FFD700",
    "VW": "#E63946",
    "EW": "#A8DADC",
    "Conjugate HF-VIX VW": "#457B9D",
    "Conjugate HF-EPU VW": "#FF69B4",
    "Conjugate HF-Climate Risk 1 VW": "#000000",
    "Conjugate HF-Climate Risk 2 VW": "#000000",
    "Conjugate HF-BMA": "#000000",
    "Jeffreys": "#1D3557",
    "Shrinkage": "#F4A261",
    "Jorion Hyperpar.": "#2A9D8F",
    "Black-Litterman": "#9370DB",
    "Greyserman Hiera.": "#9DC209",
    }

    if display_name not in colors:
        print(f"⚠ WARNING: No color assigned for {display_name}. Defaulting to black.")
        return "#000000"    


def get_display_name_from_full_name(full_name):
    if "conjugate_hf_vix_vw" in full_name:
        display_name = "Conjugate HF-VIX VW"
    elif "conjugate_hf_epu_vw" in full_name:
        display_name = "Conjugate HF-EPU VW"
    elif "conjugate_hf_climate_risk1_vw" in full_name:
        display_name = "Conjugate HF-Climate Risk 1 VW"
    elif "conjugate_hf_climate_risk2_vw" in full_name:
        display_name = "Conjugate HF-Climate Risk 2 VW"
    elif "conjugate_hf_bma" in full_name:
        display_name = "Conjugate HF-BMA"  # Ensure this is present
    elif "jeffreys" in full_name:
        display_name = "Jeffreys"
    elif "black_litterman" in full_name:
        display_name = "Black-Litterman"
    elif "shrinkage" in full_name:
        display_name = "Shrinkage"
    elif "jorion" in full_name:
        display_name = "Jorion Hyperpar."
    elif "greyserman" in full_name:
        display_name = "Greyserman Hiera."
    elif "vw" in full_name:
        display_name = "VW"
    elif "ew" in full_name:
        display_name = "EW"
    else:
        print(f"⚠ WARNING: No display name found for {full_name}")
        display_name = None

    return display_name

# use_strategies = ["vw", "ew", "conjugate_hf_vix_vw", "conjugate_hf_epu_vw", "conjugate_hf_climate_risk1_vw", "conjugate_hf_climate_risk2_vw", "conjugate_hf_bma","jeffreys",  "shrinkage", "jorion", "black_litterman", "greyserman"]

# All strategies
# use_strategies = ["vw", "ew", "conjugate_hf_bma", "jeffreys",  "shrinkage", "jorion", "black_litterman", "greyserman"]    

# Just bma for fast check
use_strategies = ["conjugate_hf_bma"]


def create_portfolio_specs():
    weighting_strategies = use_strategies
    sizes = [50] # NUmber of assets in the portfolio
    risk_aversions = [5, 10] 
    #turnover_costs = [0, 5, 10, 15, 20, 25, 30, 35] # Turnover costs
    turnover_costs = [15]
    rebalancing_frequencies = ["monthly"]
    rolling_window = [250]
    rolling_window_frequenies = ["weekly"]
    #mcm_scalings = [0.001, 1, 5, 20]
    mcm_scalings = [1] # Detemine how much market conditions (e.g. EPU, VIX) should influence the portfolio weights

    all_portfolio_specs = {}

    for weighting_strategy in weighting_strategies:
        # Determine valid risk aversions based on the weight spec
        valid_risk_aversions = [None] if weighting_strategy in {"vw", "ew"} else risk_aversions
        valid_mcm_scalings = [None] if not weighting_strategy in {"conjugate_hf_vix_vw", "conjugate_hf_epu_vw", "conjugate_hf_climate_risk1_vw", "conjugate_hf_climate_risk2_vw", "conjugate_hf_bma"} else mcm_scalings

        for size, risk, turnover, freq, window, window_freq, mcm_scaling in product(
            sizes, valid_risk_aversions, turnover_costs, rebalancing_frequencies, rolling_window, rolling_window_frequenies, valid_mcm_scalings):

            risk_label = "NA" if risk is None else risk
            mcm_scaling_label = "NA" if mcm_scaling is None else mcm_scaling

            key = f"weighting_strategy_{weighting_strategy}_size_{size}_risk_aversion_{risk_label}_turnover_cost_{turnover}_rebalancing_frequency_{freq}_rolling_window_{window}_rolling_window_frequency_{window_freq}_mcm_scaling_{mcm_scaling_label}"
            display_name = get_display_name_from_full_name(key)

            all_portfolio_specs[key] = {
                "weighting_strategy": weighting_strategy,
                "size": size,
                "risk_aversion": risk,
                "turnover_cost": turnover,
                "rebalancing_frequency": freq,
                "rolling_window": window,
                "rolling_window_frequency": window_freq,
                "mcm_scaling": mcm_scaling,
                "display_name": display_name
            }

    return all_portfolio_specs
