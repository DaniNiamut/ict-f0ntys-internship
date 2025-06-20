import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.isotonic import IsotonicRegression
from data_generation import sigmoid_fn, linear_fn, exp_fn, dec_exp_fn
import string


def plot_well_data(
    time_column : np.ndarray | list[float],
    y_true : np.ndarray | list[float],
    y_filtered : np.ndarray | list[float],
    y_fit : np.ndarray | list[float],
    norm_react_rates : np.ndarray | list[float],
    wells : list[str]
    ) -> None:
    """
    Plots the yield data, the filtered data and a fit for all wells in a grid layout.
    Also incuding the normalized reaction rates and well names each subplot.

    Parameters
    ----------
    time_column : numpy array or list of floats
        The time values corresponding to the yield data.
    
    y_true : numpy array or list of floats
        The real yield data for each well.
    
    y_filtered : numpy array or list of floats
        The filtered yield data for each well.
    
    y_fit : numpy array or list of floats
        The predicted yield data for each well.
    
    norm_react_rates : numpy array or list of floats
        The normalized reaction rates for each well.

    wells : list of str
        Names of each well formatted as 'A1', 'B2', etc.
    """
    max_row = max(w[0] for w in wells)
    max_col = max(int(w[1:]) for w in wells)

    rows = string.ascii_uppercase.index(max_row) + 1
    cols = max_col

    well_to_index = {w: i for i, w in enumerate(wells)}

    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=False)
    axes = np.array(axes).reshape((rows, cols))

    for i in range(rows):
        for j in range(cols):
            row_letter = string.ascii_uppercase[i]
            col_number = j + 1
            well_name = f"{row_letter}{col_number}"
            ax = axes[i, j]

            if well_name in well_to_index:
                idx = well_to_index[well_name]
                ax.plot(time_column, y_true[:, idx], label='Raw', color="#FF7700")      # orange
                ax.plot(time_column, y_filtered[:, idx], '--',  label='Filtered', color='#4CAF50', alpha=0.5)  # green
                ax.plot(time_column, y_fit[:, idx], '-', label='Fit', color='#257BB6', alpha=0.5)       # blue

                ax.text(0.1, 0.9, well_name, transform=ax.transAxes,
                        fontsize=10, fontweight='bold', va='top', ha='left')
                ax.text(0.1, 0.70, f"{norm_react_rates[idx]:.3f}", transform=ax.transAxes,
                        fontsize=10, va='top', ha='left', color='red')
            else:
                ax.set_visible(False)

            ax.tick_params(labelsize=6)

    fig.suptitle("Yield over Time for all wells")
    fig.supylabel("Yield")
    fig.supxlabel("Time (s)")

    fig.text(0.78, 0.975, "Legend:", color='black')
    fig.text(0.84, 0.98, "Raw", fontweight='bold', color='#FF7F0E')
    fig.text(0.84, 0.96, "Filtered", fontweight='bold', color='#4CAF50')
    fig.text(0.91, 0.96, "Fit", fontweight='bold', color='#257BB6')
    fig.text(0.91, 0.98, "Reaction Rate", fontweight='bold', color='red')


    plt.tight_layout()
    plt.show()

def least_squares_fitter(t_vals, y_vals) -> tuple[dict, list[float]]:
    """
    Fits the given time and yield values to several models: sigmoid, linear, exponential, and decaying exponential.

    Parameters
    ----------
    t_vals : Numpy array or List of Floats
        The time values corresponding to the yield data.

    y_vals : Numpy array or List of Floats
        The yield values to fit against the time values.

    Returns
    -------
    params : Dictionary
        A dictionary containing the parameters of the fitted models.
        Keys are 'sigmoid', 'linear', 'exp', and 'dec exp' with their respective parameters as tuples.
    
    scores : List of floats
        A list of R-squared scores for each fitted model, indicating the goodness of fit.
        The order of scores corresponds to the order of models in the params dictionary whuch is:
        ['sigmoid', 'linear', 'exp', 'dec exp'].
    """
    t_vals = np.asarray(t_vals).reshape(-1)
    y_vals = np.asarray(y_vals).reshape(-1)

    params = {}
    scores = []

# Sigmoid approximation
    L = np.max(y_vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        safe_vals = np.clip(y_vals, 1e-5, None)
        y_ratio = L / safe_vals - 1
        y_ratio = np.clip(y_ratio, 1e-10, None)
        z = np.log(y_ratio)
    # Estimate t0 as the t at max slope
    slopes = np.gradient(y_vals, t_vals)
    t0_index = np.argmax(slopes)
    t0 = t_vals[t0_index]
    X = (t_vals - t0).reshape(-1, 1)

    model = LinearRegression(fit_intercept=False).fit(X, z)
    k = model.coef_[0]
    params['sigmoid'] = (L, k, t0)
    y_pred = sigmoid_fn(t_vals, *params['sigmoid'])
    sigmoid_score = r2_score(y_vals, np.clip(y_pred, -1e10, 1e10))
    scores.append(sigmoid_score)

    # Linear
    X = t_vals.reshape(-1, 1)
    model = LinearRegression().fit(X, y_vals)
    coef, intercept = model.coef_[0], model.intercept_
    params['linear'] = (coef, intercept)
    y_pred = linear_fn(t_vals, *params['linear'])
    linear_score = r2_score(y_vals, np.clip(y_pred, -1e10, 1e10))
    scores.append(linear_score)

    # Exponential
    with np.errstate(divide='ignore'):
        z = np.log(np.clip(y_vals, 1e-10, None))
    model = LinearRegression().fit(X, z)
    b = np.exp(model.intercept_)
    a = model.coef_[0]
    params['exp'] = (a, b)  # y = a * exp(b*t)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        raw = exp_fn(t_vals, *params['exp'])
    y_pred = np.clip(raw, -1e10, 1e10)
    y_pred = np.nan_to_num(y_pred)
    exp_score = r2_score(y_vals, y_pred)
    scores.append(exp_score)

    # Negative Exponential
    a = np.max(y_vals)
    d = np.min(y_vals)
    slopes = np.gradient(y_vals, t_vals)
    peak_index = np.argmax(slopes)
    c = t_vals[peak_index]

    y_tail = y_vals[peak_index:]
    t_tail = t_vals[peak_index:]
    with np.errstate(divide='ignore'):
        z = np.log(np.clip(a - y_tail, 1e-10, None))
    X_tail = (t_tail - c).reshape(-1, 1)

    model = LinearRegression().fit(X_tail, z)
    b = -model.coef_[0]
    params['dec exp'] = (a, b, c, d)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        y_pred = dec_exp_fn(t_vals, *params['dec exp'])
    y_pred = np.nan_to_num(y_pred)
    dec_exp_score = r2_score(y_vals, np.clip(y_pred, -1e10, 1e10))
    scores.append(dec_exp_score)

    return params, scores

def preprocessor(
    settings_df : pd.DataFrame,
    raw_df : pd.DataFrame,
    pos_wells : list[str] | None = None,
    override_wells: list[str] | None = None,
    plot : bool = False,
    return_coef : bool = True,
    return_function_type : bool = True
    ) -> pd.DataFrame:
    # Only keep linear and decaying exponential parameters
    """
    Preprocesses the raw yield data from a DataFrame, fits models to the data, and returns a DataFrame with the results.

    Parameters
    ----------
    settings_df : pandas DataFrame
        DataFrame containing settings for each well, indexed by well names.

    raw_df : pandas DataFrame
        DataFrame containing the original data.

    pos_wells : list of strings
        List of well names to use for normalization of reaction rates. If None, all wells are used.
    
    override_wells : list of strings
        List of well names to override the the wells that we dont want to take into consderation DataFrame.
        If None, no override is performed. (Currently not implemented)

    plot : bool
        If True, plots the yield data, filtered data, and fitted models for each well.

    return_coef : bool
        If True, includes the fitted coefficients in the returned DataFrame.

    return_function_type : bool
        If True, includes the type of function used for fitting in the returned DataFrame.

    Returns
    -------
    pandas DataFrame
        A DataFrame containing the processed data with the following columns:
        - 'well': Well names.
        - 'norm_yield_grad': Normalized reaction rates.
        - 'max_yield': Maximum yield for each well.
        - 'function_type': Type of function used for fitting (if return_function_type is True).
        - Coefficients of the fitted models (if return_coef is True).
    """
    params_names = ['linear_a', 'linear_b', 'dec_exp_a', 'dec_exp_b', 'dec_exp_c', 'dec_exp_d']
    params_indices = {
        'linear': [0, 1],
        'dec exp': [2, 3, 4, 5]
    }
    params_functs = {
        'linear': linear_fn,
        'dec exp': dec_exp_fn
    }

    cols = list(raw_df.columns)
    wells = cols[2:]
    max_react_rates = []
    yields = []
    best_fits = []
    all_params = []
    all_params_lite = []
    x_fits = []

    settings_df = settings_df.set_index('well').T.reset_index()
    settings_df.columns.name = None
    settings_df = settings_df.drop(columns='index')

    if np.issubdtype(raw_df['Time'].dtype, np.number):
        time_column = raw_df['Time'].to_numpy()
    else:
        time_column = raw_df['Time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second).to_numpy()

    #time_column = raw_df['Time'].apply(lambda t: (t.hour * 3600 + t.minute * 60 + t.second)).to_numpy()

    for well in wells:
        params_as_vars = np.zeros(6)  # Updated to match new param count
        y_vals = raw_df[well].to_numpy()
        ir = IsotonicRegression(increasing=True)
        x_fit = ir.fit_transform(time_column, y_vals)
        total_yield = max(y_vals)
        slopes = np.diff(x_fit) / np.diff(time_column)
        t0 = max(slopes)

        # Fit and filter only linear and dec_exp
        params, scores = least_squares_fitter(time_column, x_fit)
        filtered_params = {k: v for k, v in params.items() if k in ['linear', 'dec exp']}
        filtered_scores = [scores[i] for i, k in enumerate(params.keys()) if k in ['linear', 'dec exp']]
        best_fit = list(filtered_params.keys())[np.argmax(filtered_scores)]

        for i, val in zip(params_indices[best_fit], filtered_params[best_fit]):
            params_as_vars[i] = val
        max_react_rates.append(t0)
        yields.append(total_yield)
        best_fits.append(best_fit)
        all_params.append(params_as_vars)
        all_params_lite.append(filtered_params[best_fit])
        x_fits.append(x_fit)

    max_react_rates = np.array(max_react_rates)
    if pos_wells:
        pos_well_ind = [i for i, col in enumerate(wells) if col in pos_wells]
        pos_wells_t0 = max_react_rates[pos_well_ind].mean()
        norm_react_rates = max_react_rates / pos_wells_t0
    else:
        norm_react_rates = max_react_rates

    if return_coef:
        settings_df[params_names] = all_params
    settings_df['norm_yield_grad'] = norm_react_rates
    settings_df['max_yield'] = yields
    if return_function_type is True:
        settings_df['function_type'] = best_fits

    if plot:
        y_fit = np.array([params_functs[best_fits[i]](time_column, *all_params_lite[i])
                           for i in range(len(best_fits))]).T
        y_true = raw_df[wells].to_numpy()
        y_filtered = np.array(x_fits).T
        plot_well_data(time_column, y_true, y_filtered, y_fit, norm_react_rates, wells)
    return settings_df
