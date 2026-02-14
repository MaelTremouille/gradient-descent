# Gradient Descent from Scratch

A from-scratch implementation of gradient descent and its variants, built with nothing but **NumPy**. No autograd, no frameworks — just the math.

## What's Inside

| Section | Description |
|---|---|
| **Numerical Differentiation** | Forward finite differences, tangent line visualization |
| **Gradient Computation** | N-dimensional gradient via partial derivatives, verified against analytical solutions |
| **Gradient Descent** | Full implementation with 4 learning rate schedules (constant, linear, quadratic, exponential decay) |
| **Multi-start GD** | Run GD from multiple random starting points to escape local minima |
| **Momentum** | Velocity-based acceleration for faster convergence in narrow valleys |
| **Stochastic Gradient Descent** | Mini-batch SGD with configurable batch size and epoch-based training |
| **Linear Regression** | Fit a line to noisy data using all three methods, with side-by-side comparison |

## Highlights

- Contour plots of gradient descent navigating the **Rosenbrock function**
- Side-by-side comparison of **Vanilla GD vs Momentum**
- **Learning rate schedule** comparison on the same problem
- Full **convergence curves** (log scale) for every experiment
- **Linear regression** solved three ways: full-batch GD, multi-start, and SGD

## Quick Start

```bash
# Clone & setup
git clone https://github.com/<your-username>/gradient-descent.git
cd gradient-descent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the notebook
jupyter notebook GD.ipynb
```

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

## Project Structure

```
gradient-descent/
├── GD.ipynb           # Main notebook — all code, math, and visualizations
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

## Math at a Glance

**Gradient descent update rule:**

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \alpha_k \nabla f(\mathbf{x}_k)$$

**Momentum update:**

$$\mathbf{v}_{k+1} = \beta \mathbf{v}_k + \nabla f(\mathbf{x}_k) \qquad \mathbf{x}_{k+1} = \mathbf{x}_k - \alpha \mathbf{v}_{k+1}$$

**SGD approximation:**

$$\nabla L(\theta) \approx \frac{1}{|B|}\sum_{i \in B} \nabla \ell_i(\theta)$$

## License

MIT
