# Heston-Pricer

A **Heston pricing calculator** for valuing vanilla and exotic options on equity.

## Overview

This repository implements the Heston stochastic volatility model, a widely-used framework in quantitative finance for pricing equity derivatives. The model efficiently prices vanilla options (standard calls and puts) as well as more complex exotic options.

## Main Python Classes

### `HestonModel` (in `heston_model.py`)
Represents the Heston stochastic volatility model. This class provides:
- Construction and storage of model parameters (`kappa`, `theta`, `xi`, `rho`, `v0`).
- Simulation of underlying asset price and volatility paths.
- Calculation of the characteristic function for the asset price distribution.
- Analytical and numerical pricing of European call options using several approaches: direct integration, Fast Fourier Transform (Carr-Madan), and Monte Carlo simulation.

#### Key Methods:
- `simulate(S0, T, r, q, npaths, nsteps, seed)`: Simulate paths for asset price and variance.
- `heston_cf(u, T, S0, r, q)`: Compute the characteristic function under the Heston model.
- `heston_call(...)`, `carr_madan_call(...)`, `monte_carlo_call(...)`: Price European options using different techniques.

### `Pricer` (in `heston_pricer.py`)
A wrapper class for pricing vanilla and exotic options using a supplied stochastic volatility model, typically an instance of `HestonModel`. This class provides:
- High-level methods to price European, digital, and barrier options.
- Integration with model calibration and simulation routines.

#### Key Methods:
- `european(T, S0, r, q, K, type)`: Price standard European options.
- `digital(T, S0, r, q, K, type, npaths)`: Price digital (binary) options via simulation.
- `barrier(T, S0, r, q, K, B, type, npaths, nsteps, seed)`: Price barrier options (knock-in/knock-out) using path simulation.

### `calibrate` Function (in `calibration.py`)
The main calibration routine for fitting the Heston model parameters to market option data.
- Uses market implied volatilities (from a volatility surface) and converts them to prices via Black-Scholes.
- Fits the Heston model parameters (`kappa`, `theta`, `xi`, `rho`, `v0`) by minimizing the squared difference between market prices and Heston model prices produced by the Carr-Madan FFT method.
- Automatically checks the Feller condition for model validity.
- Provides detailed output about the calibration process, including parameter values, error, and price differences.

#### Key Methods/Functions:
- `calibrate(heston_model, df_surf, S0, r, T, q, ...)`: Fits model parameters to market data for a single maturity.
- `bs_call_price(...)`: Computes Black-Scholes prices for conversion from vols to prices.
- `get_vol_slice(...)`: Extracts and interpolates implied volatilities for a given maturity.

## Features

- Fast and accurate pricing for vanilla options (calls, puts)
- Support for exotic option structures (digital, barrier)
- Parameter calibration to market data using the Carr-Madan approach.

## Example Usage

See `Example.ipynb` for demonstrations on model simulation, calibration, and option pricing.

## Installation

Clone the repository:
```bash
git clone https://github.com/MarchesiQuant/Heston-Pricer.git
cd Heston-Pricer
```

*Created and maintained by [MarchesiQuant](https://github.com/MarchesiQuant)*
