def yield_time_x(x, t, params, tau, mu):
    w = _weights(x, tau, mu)
    a1, b1, c1 = params['sigmoid']
    a2, b2 = params['linear']
    a3, b3 = params['exp']
    a4, b4 = params['log']
    
    return (w[0] * sigmoid_fn(t, a1, b1, c1) +
            w[1] * linear_fn(t, a2, b2) +
            w[2] * exp_fn(t, a3, b3) +
            w[3] * log_fn(t, a4, b4))

def faithful_yield_time_x(x_scalar, t, params, tau, mu):
    x_scalar = np.asarray(x_scalar)  # shape (n_obs,)
    weights = _weights(x_scalar, tau, mu)  # shape (n_funcs, n_obs)
    n_obs = x_scalar.shape[0]
    yield_val = np.zeros((n_obs, len(t)))

    all_params = params['linear'] + params['dec exp']

    for i, param in enumerate(all_params):
        w_i = weights[i]  # shape (n_obs,)
        if i < len(params['linear']):
            a, b = param
            g_i = linear_fn(t, a, b)  # shape (len(t),)
        else:
            a, b, c, d = param
            g_i = dec_exp_fn(t, a, b, c, d)
        yield_val += w_i[:, None] * g_i[None, :]

    return yield_val