import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.interpolate import interp1d


def calibrate(heston_model, df_surf, S0, r, T, q = 0, alpha=1.5, N_fft=4096, eta=0.25):
    """
    Calibrate Heston model parameters to market implied vols for a single maturity.
    Market vols are extracted internally from df_surf and converted to call prices via Black-Scholes.
    """
    
    # Extract strikes and vol and convert to prices 
    K_market, sigma_market = get_vol_slice(df_surf, T)
    C_market = bs_call_price(S0, K_market, T, r, q, sigma_market)

    # Initial guess and bounds: [kappa, xi, rho, v0, phi]
    x0 = [0.5, 0.3, -0.5, 0.1, 0.2]  
    bounds = [(1e-3, 10), (1e-3, 2), (-0.99, 0.99), (1e-3, 1), (0, 1)]

    K_market = np.atleast_1d(K_market)
    C_market = np.atleast_1d(C_market)
    history = []

    # Objective function for calibration
    def objective(params):
        kappa, xi, rho, v0, phi = params
        theta = (xi**2 + phi) / (2 * kappa)  # Feller guaranteed

        heston_model.kappa = kappa
        heston_model.theta = theta
        heston_model.xi = xi
        heston_model.rho = rho
        heston_model.v0 = v0

        C_model = heston_model.carr_madan_call(T, S0, r, q, K_market, alpha=alpha, N=N_fft, eta=eta)
        error = np.sum((C_model - C_market)**2)   
        history.append((params, error))
        return error

    # Minimization 
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', options={'disp': True, 'maxiter': 500})

    # Calibration results 
    if result.success:
        kappa, xi, rho, v0, phi = result.x
        theta = (xi**2 + phi) / (2 * kappa)
        heston_model.kappa = kappa
        heston_model.theta = theta
        heston_model.xi = xi
        heston_model.rho = rho
        heston_model.v0 = v0

        print(f"Calibration successful:")
        print(f"Iterations: {result.nit}")
        print(f"Total Error: {result.fun:.5e}.")
        print("\nParameters:")
        print(f"kappa: {kappa:.5f}, theta: {theta:.5f}, xi: {xi:.5f}, "f"rho: {rho:.5f}, v0: {v0:.5f}\n")

        # Feller condition
        if 2 * kappa * theta < xi**2: print("Warning: Feller condition NOT satisfied!\n")

        # Compare model prices vs market prices
        C_model = heston_model.carr_madan_call(T, S0, r, q, K_market, alpha=alpha, N=N_fft, eta=eta)
        print("Market vs Model prices differences (%):")
        for k, cm, cm_model in zip(K_market, C_market, C_model):
            dif = (cm_model - cm)/cm * 100
            print(f"Moneyness {100 * k/S0:.2f}%: Model vs Market difference: {dif:.4f}%")
    else:
        print("Calibration failed:", result.message)

    return result



def bs_call_price(S0, K, T, r, q, sigma):
    """
    Black-Scholes price of a European call with continuous dividend yield.
    """
    K = np.atleast_1d(K)
    sigma = np.atleast_1d(sigma)
    F = S0 * np.exp((r - q) * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return call_price


def get_vol_slice(df_vol, T):
    """
    Extract strikes and implied volatilities for a given maturity T.

    Parameters
    ----------
    df_vol : pd.DataFrame
        DataFrame with index = maturities (years), columns = strikes + 'ATM'
        The first column can be 'Date' or maturities.
    T : float
        Expiry for which we want the vol slice
    S0 : float
        Spot price to replace 'ATM'
    
    Returns
    -------
    K_market : np.ndarray
        Array of strikes (with ATM replaced by S0)
    sigma_market : np.ndarray
        Array of interpolated volatilities for T
    """
    
    df = df_vol.copy()
    
    # Select only numeric columns (ignore first column like 'Date')
    numeric_cols = [col for col in df.columns if col not in ['Date', 'ATM']]
    df_numeric = df[numeric_cols]

    # Get maturities and strikes 
    maturities = df['Date'].values.astype(float)
    K_market = df_numeric.columns.astype(float).to_numpy()
    sigma_market = []

    # Interpolate vols
    for k in numeric_cols:
        vols_k = df_numeric[k].values.astype(float)
        f = interp1d(maturities, vols_k, kind='linear', fill_value="extrapolate")
        sigma_market.append(f(T))

    sigma_market = np.array(sigma_market)/100

    # Sort by strike
    sort_idx = np.argsort(K_market)
    return K_market[sort_idx], sigma_market[sort_idx]