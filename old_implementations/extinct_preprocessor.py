#preprocessor but with sigmoid, linear, exp and dec exo
from botorch.sampling import SobolQMCNormalSampler
from data_generation import linear_fn, sigmoid_fn, exp_fn, dec_exp_fn
def preprocessor(settings_df, raw_df, pos_wells, override_wells=None, plot=False, return_coef=False):
    params_names = ['sigmoid_a', 'sigmoid_b', 'sigmoid_c', 'linear_a', 'linear_b', 'exp_a',
               'exp_b', 'dec_exp_a', 'dec_exp_b', 'dec_exp_c', 'dec_exp_d']
    params_indices = {'sigmoid': [0, 1, 2],
        'linear': [3, 4],
        'exp': [5, -5],
        'dec exp': [-4, -3, -2, -1]}
    params_functs = {'sigmoid': sigmoid_fn,
        'linear': linear_fn,
        'exp': exp_fn,
        'dec exp': dec_exp_fn}

    cols = list(raw_df.columns)
    wells = cols[2:]
    max_react_rates = []
    yields = []
    best_fits = []
    params = {}
    all_params = []
    all_params_lite = []
    x_fits = []

    settings_df = settings_df.set_index('well').T.reset_index()
    settings_df.columns.name = None
    settings_df = settings_df.drop(columns='index')
    time_column = raw_df['Time'].apply(lambda t: (t.hour * 3600 + t.minute * 60 + t.second)).to_numpy()

    for well in wells:
        params_as_vars = np.zeros(11)
        y_vals = raw_df[well].to_numpy()

        ir = IsotonicRegression(increasing=True)
        x_fit = ir.fit_transform(time_column, y_vals)
        total_yield = max(y_vals)
        slopes = np.diff(x_fit) / np.diff(time_column)
        t0 = max(slopes)

        params, scores = least_squares_fitter(time_column, x_fit)
        best_fit = list(params.keys())[np.argmax(scores)]
        params_as_vars[params_indices[best_fit]] = params[best_fit]

        max_react_rates.append(t0)
        yields.append(total_yield)
        best_fits.append(best_fit)
        all_params.append(params_as_vars)
        all_params_lite.append(params[best_fit])
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
    settings_df['function_type'] = best_fits

    if plot:
        y_fit = np.array([params_functs[best_fits[i]](time_column, *all_params_lite[i])
                           for i in range(len(best_fits))]).T
        y_true = raw_df[wells].to_numpy()
        y_filtered = np.array(x_fits).T
        plot_well_data(time_column, y_true, y_filtered, y_fit, norm_react_rates, wells)
    return settings_df
