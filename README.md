# Deep Energy Method for 2D Elastic and Neo-Hookean Solid Mechanics

A Physics-Informed Machine Learning (PIML) implementation of the Deep Energy Method (DEM) for solving two-dimensional elasticity problems using neural networks and energy minimization.

This project investigates the application of neural networks to computational solid mechanics by approximating displacement fields through minimization of the total potential energy. Both linear elastic and compressible Neo-Hookean material models are considered and validated against analytical solutions and FEM reference simulations. 【1-216556】【2-9a7514】

---

## Overview

The project solves a two-dimensional tensile test problem:

- Rectangular domain
- Fixed left boundary (Dirichlet condition)
- Applied traction on the right boundary (Neumann condition)
- Plane stress/plane strain solid mechanics formulation
- Physics-informed training via energy minimization

Unlike conventional Finite Element Methods (FEM), the displacement field is represented by a neural network and trained directly from physical principles without requiring a full labeled dataset. 【2-9a7514】

---

## Motivation

Physics-Informed Machine Learning offers an alternative approach to classical numerical methods by embedding governing equations and physical constraints directly into the training process.

This project explores:

- Deep Energy Methods (DEM)
- Physics-Informed Neural Networks (PINNs)
- Data-driven pretraining
- Transfer learning between data-driven and physics-based optimization
- Hyperparameter studies
- Advanced material modeling
- Transformer architectures for computational mechanics 【2-9a7514】

---

## Physical Problem

The considered benchmark problem consists of a rectangular specimen subjected to uniaxial tension.

```text
Fixed Boundary                  Applied Traction

|                               →
|                               →
|                               →
|_______________________________→
```

### Geometry

```text
Lx = 2
Ly = 1
```

### Material Parameters

```text
Young's Modulus E = 1
Poisson Ratio ν = 0.3
```

### Boundary Conditions

#### Dirichlet Boundary

```text
u = 0
```

on the left edge.

#### Neumann Boundary

```text
T = 0.2
```

on the right edge.

#### Free Boundaries

Upper and lower boundaries are traction-free. 【1-216556】【2-9a7514】

---

## Methodology

### Deep Energy Method

Instead of solving equilibrium equations directly, the displacement field is obtained through minimization of the total potential energy:

```math
\Pi = \Psi - T
```

where

- Ψ = internal strain energy
- T = external traction work

The neural network represents the displacement field

```math
u_\theta(x,y)
```

and is trained to satisfy energy equilibrium. 【2-9a7514】

---

## Material Models

### Linear Elasticity

The linear elastic formulation is based on Hooke's law:

```math
\sigma = 2\mu\varepsilon + \lambda \, tr(\varepsilon) I
```

with Lamé parameters

```math
\mu = \frac{E}{2(1+\nu)}
```

```math
\lambda = \frac{E\nu}{(1+\nu)(1-2\nu)}
```

### Compressible Neo-Hookean Material

The project additionally implements a hyperelastic Neo-Hookean constitutive model based on:

```math
\Psi =
\frac{\lambda}{2}(\ln J)^2
-
\mu \ln J
+
\frac{\mu}{2}(I_1-2)
```

allowing finite deformation simulations and comparison with FEM reference results. 【2-9a7514】

---

## Neural Network Architecture

The displacement field

```math
u(x,y) = [u_x,u_y]
```

is approximated by a fully connected neural network.

### Inputs

- x-coordinate
- y-coordinate

### Outputs

- horizontal displacement uₓ
- vertical displacement uᵧ

### Network Configuration

```text
Input Layer (2)
        ↓
Dense(16)
        ↓
Custom Activation
        ↓
Dense(2)
```

The implementation includes a custom trainable activation function:

```python
f(x) = x² + αx
```

where α is learned during training. 【1-216556】

---

## Loss Function

The total loss combines multiple physics-based objectives.

### Energy Residual

```math
(T-\Psi)^2
```

Ensures consistency between external work and internal strain energy.

### Boundary Condition Loss

Enforces:

- fixed displacement boundary
- traction boundary conditions

### Divergence Loss

```math
\nabla \cdot \sigma = 0
```

Promotes satisfaction of local equilibrium equations.

### Data-Driven Loss

An analytically approximated displacement field can be used for initial supervised pretraining before switching to physics-informed optimization. 【1-216556】【2-9a7514】

---

## Training Strategy

The project employs a hybrid learning approach.

### Phase 1

Data-driven pretraining:

```text
MSE(Upred, Utarget)
```

### Phase 2

Physics-informed transfer learning:

```text
(T - Ψ)^2
+ Boundary Loss
+ Divergence Loss
```

This strategy significantly improves convergence and solution quality. 【2-9a7514】

---

## Numerical Features

### Automatic Differentiation

PyTorch autograd is used for:

- strain computation
- deformation gradients
- stress evaluation
- divergence calculation

### Numerical Integration

Implemented integration schemes:

- Trapezoidal Rule
- Composite Simpson Rule

A dedicated study demonstrates improved accuracy when Simpson integration is used. 【1-216556】【2-9a7514】

---

## Hyperparameter Studies

The report investigates the influence of:

- Number of epochs
- Boundary loss formulation
- Divergence loss activation
- Data-driven pretraining
- Traction energy calculation methods
- Incremental force application
- Integration rules
- Optimizer choice (Adam vs LBFGS)
- Network architecture
- Number of collocation points

and analyzes their impact on convergence, deformation prediction and energy consistency. 【2-9a7514】

---

## Results

The developed DEM framework successfully:

✅ predicts physically meaningful displacement fields

✅ reproduces tensile deformation behavior

✅ approximates FEM strain energy values

✅ supports hyperelastic Neo-Hookean materials

✅ benefits from transfer learning and incremental loading

✅ demonstrates convergence of energy-based loss functions

For Neo-Hookean materials, qualitative agreement with Abaqus FEM simulations is achieved regarding deformation fields and strain energy evolution. 【2-9a7514】

---

## Experimental Extensions

Beyond the Deep Energy Method, this project also investigates:

### Vision Transformers (ViT)

Adaptation of transformer architectures to regression and mechanics tasks.

Features include:

- patch-based mesh encoding
- multi-head self-attention
- physics-informed training objectives

### Graph Transformers

Conceptual exploration of Energy-Informed Graph Transformer architectures for future mesh-free computational mechanics applications. 【2-9a7514】

---

## Installation

```bash
git clone https://github.com/yourusername/deep-energy-method-solid-mechanics.git

cd deep-energy-method-solid-mechanics
```

Install dependencies:

```bash
pip install torch numpy matplotlib
```

---

## Usage

Run the main script:

```bash
python DEM_pltTraction_datadriven_neo_hooke_2.py
```

The script will:

1. Generate collocation points
2. Build the neural network
3. Perform data-driven pretraining
4. Switch to physics-informed training
5. Compute energy quantities
6. Evaluate tractions
7. Visualize 
