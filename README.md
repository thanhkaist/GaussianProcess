# GaussianProcess

A clean, from-scratch Python implementation of **Gaussian Process Regression (GPR)** using NumPy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Mathematical Background](#mathematical-background)
- [Repository Structure](#repository-structure)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

Gaussian Processes (GPs) are a powerful class of probabilistic, non-parametric models widely used for regression and uncertainty quantification. Rather than learning a fixed set of parameters, a GP defines a distribution over functions and updates it as training data are observed, giving both a mean prediction and a principled measure of uncertainty at every test point.

This repository provides a minimal, readable implementation to help understand the mechanics of GP regression without the abstraction layers of large ML frameworks.

---

## Features

- Zero-mean GP prior with a **squared-exponential (RBF) kernel**
- Numerically stable training via **Cholesky decomposition**
- Posterior **mean** and **standard deviation** predictions
- Easy-to-follow class-based (`GausianProcessRegression`) and script-based (`gp1.py`) implementations
- Visualization of prior samples, posterior samples, and confidence bands

---

## Installation

### From source

```bash
git clone https://github.com/thanhkaist/GaussianProcess.git
cd GaussianProcess
pip install -e .
```

### Dependencies only

```bash
pip install numpy matplotlib
```

---

## Quick Start

```python
import numpy as np
from gp.gp import GausianProcessRegression

# 1. Define training data
N = 10
s = 0.00005                                  # noise variance
X_train = np.random.uniform(-5, 5, (N, 1))
Y_train = np.sin(0.9 * X_train).flatten() + s * np.random.randn(N)

# 2. Fit the model
gp = GausianProcessRegression()
gp.fit(X_train, Y_train)

# 3. Predict on test points
X_test = np.linspace(-5, 5, 50).reshape(-1, 1)
mu, sigma = gp.predict(X_test)

print("Predicted mean:   ", mu[:5])
print("Predicted std dev:", sigma[:5])
```

Running `gp/gp.py` directly also generates and saves two plots:

| File | Description |
|------|-------------|
| `predictive.png` | Mean prediction ± 3 standard deviations against training points |
| `prior.png` | Ten random samples drawn from the GP prior |

```bash
python gp/gp.py
```

---

## API Reference

### `gp.gp.GausianProcessRegression`

#### `__init__(kernel_=None)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kernel_` | callable or `None` | `None` | Kernel function `k(a, b) -> ndarray`. Defaults to the built-in squared-exponential kernel with `σ = 0.1`. |

#### `fit(X, Y)`

Train the model by computing the Cholesky factor of the covariance matrix.

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `X` | `(N, D)` | Training inputs |
| `Y` | `(N,)` | Training targets |

#### `predict(Xtest) -> (mu, sigma)`

Return the posterior mean and standard deviation at test locations.

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `Xtest` | `(M, D)` | Test inputs |

**Returns**

| Name | Shape | Description |
|------|-------|-------------|
| `mu` | `(M,)` | Posterior mean |
| `sigma` | `(M,)` | Posterior standard deviation |

### Kernel

The default squared-exponential (RBF) kernel is:

```
k(a, b) = exp( -0.5 / σ * ||a - b||² )
```

where `σ = 0.1` controls the length-scale.

---

## Mathematical Background

Given training data `(X, y)` and test inputs `X*`, the GP posterior is:

```
μ*  = K(X*, X) [K(X, X) + σ²I]⁻¹ y
Σ*  = K(X*, X*) − K(X*, X) [K(X, X) + σ²I]⁻¹ K(X, X*)
```

where `K(·, ·)` is the kernel (covariance) matrix. Computing the matrix inverse directly is expensive and numerically unstable; this implementation uses the **Cholesky decomposition** `L Lᵀ = K(X, X) + σ²I` and solves triangular systems instead.

For a thorough tutorial, see:
- [Gaussian Processes for Dummies – Kat Bailey](https://katbailey.github.io/post/gaussian-processes-for-dummies/)
- [Gaussian Processes for Machine Learning – Rasmussen & Williams (free PDF)](http://www.gaussianprocess.org/gpml/)

---

## Repository Structure

```
GaussianProcess/
├── gp/
│   ├── __init__.py
│   ├── gp.py          # Class-based GPR implementation + runnable example
│   └── gp1.py         # Script-based GPR (prior & posterior plots)
├── examples/          # Additional example notebooks / scripts
├── doc/               # Documentation assets
├── test/              # Unit tests
├── setup.py
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Linear algebra (Cholesky, matrix solves) |
| `matplotlib` | Plotting predictions and samples |

Install with:

```bash
pip install numpy matplotlib
```

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
