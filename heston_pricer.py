import numpy as np


class Pricer:
    """
    Pricer class for vanilla and exotic options using the Heston model.
    """

    def __init__(self, model):
        """
        Initialize the Pricer with a calibrated model.

        Parameters
        ----------
        model : object
            A model instance (e.g., Heston, Jump-Heston) that implements pricing and simulation methods.
        """
        self.model = model


    def european(self, T, S0, r, q, K, type='Call'):
        """
        Price European options using the Heston model.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        T : float
            Time to maturity (in years).            
        r : float
            Risk-free rate.
        q : float
            Dividend yield. 
        K : float or ndarray
            Strike(s) for which to return prices.
        type : str, optional
            Option type: 'Call' or 'Put'. Default is 'Call'.
        npaths : int, optional
            Number of Monte Carlo paths (relevant if model method uses Monte Carlo internally).

        Returns
        -------
        price : float or ndarray
            Option price corresponding to strike K.
        """
        call_price = self.model.heston_call(T, S0, r, q, K)
        price = call_price if type == 'Call' else call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
        return price[0]


    def digital(self, T, S0, r, q, K, type='Call', npaths=10**5):
        """
        Price European digital options using Monte Carlo simulation under the Heston model.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        T : float
            Time to maturity (in years).
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        K : float or ndarray
            Strike.
        type : str, optional
            Option type: 'Call' or 'Put'. Default is 'Call'.
        npaths : int, optional
            Number of Monte Carlo paths to simulate.

        Returns
        -------
        price : float
            Digital option price corresponding to strike K.
        """
        S_t, _ = self.model.simulate(S0, T, r, q, npaths=npaths, nsteps=252)
        S_T = S_t[-1, :]
        price = np.exp(-r * T) * np.mean(S_T > K) if type == 'Call' else np.exp(-r * T) * np.mean(S_T < K)
        return price


    def barrier(self, T, S0, r, q, K, B, type='UpAndOut', npaths=10**5, nsteps=252, seed=None):
        """
        Price European barrier options using Monte Carlo simulation under the Heston model.

        Parameters
        ----------
        S0 : float
            Initial spot price.
        T : float
            Time to maturity (in years).
        r : float
            Risk-free rate.
        q : float
            Dividend yield.
        K : float or ndarray
            Strike.
        B : float
            Barrier level.
        type : str, optional
            Option type: 'UpAndOut', 'DownAndOut', 'UpAndIn', 'DownAndIn'. Default is 'UpAndOut'.
        npaths : int, optional
            Number of Monte Carlo paths to simulate.
        nsteps : int, optional
            Number of time steps per path.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        price : float
            Barrier option price corresponding to strike K.
        """
        S_t, _ = self.model.simulate(S0, T, r, q, npaths=npaths, nsteps=nsteps, seed=seed)
        knocked_out = np.zeros(npaths, dtype=bool)

        if type in ['UpAndOut', 'UpAndIn']:
            knocked_out = np.any(S_t >= B, axis=0)
        elif type in ['DownAndOut', 'DownAndIn']:
            knocked_out = np.any(S_t <= B, axis=0)
        else:
            raise ValueError(
                "Invalid barrier option type. Choose from 'UpAndOut', 'DownAndOut', 'UpAndIn', 'DownAndIn'.")

        payoff = np.maximum(S_t[-1, :] - K, 0)

        if type.endswith('Out'):
            payoff[knocked_out] = 0  # knockout
        else:
            payoff[~knocked_out] = 0  # knock-in

        price = np.exp(-r * T) * np.mean(payoff)
        return price
