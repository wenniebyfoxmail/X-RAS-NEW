# X-RAS-NEW

Extended residual-adaptive physics-informed neural network experiments for phase-field fracture prediction.

This repository contains a research prototype for learning fracture and damage evolution with physics-informed neural networks. It explores a two-stage workflow that starts from a baseline phase-field PINN and extends it with domain decomposition, interface consistency, and residual-based adaptive sampling around the crack region.

## Research Motivation

Physics-informed neural networks are attractive for mechanics problems because they can encode governing equations and physical constraints. Fracture problems are especially challenging because the solution develops sharp localised damage, strong gradients, and crack-tip singular behaviour.

This project investigates whether domain decomposition and adaptive sampling can improve PINN reliability for phase-field fracture prediction.

## Method Overview

The X-RAS-PINN workflow combines:

- a displacement network and damage network
- phase-field fracture energy terms
- domain partitioning into crack-near and far-field regions
- interface losses for consistency between subdomains
- residual/adaptive sampling guided by strain energy density and damage gradients
- ablation experiments for partitioning, interface loss, and adaptive sampling

## Repository Structure

- `solver_pinn.py` - baseline phase-field PINN components
- `solver_xras.py` - X-RAS-PINN domain-decomposition and adaptive-sampling solver
- `config.py` - shared configuration for debug and full experiments
- `Run_experiments.py` - experiment runner for baseline, ablation, and parameter sweeps
- `test_sent_pinn.py` - baseline SENT-with-notch experiment
- `test_sent_xras.py` - X-RAS phase-2 experiments
- `phase1_phase2_bridge.py` - workflow bridge between training stages
- `outputs/` - generated checkpoints and comparison figures

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib

python Run_experiments.py --exp phase1 --mode debug
python Run_experiments.py --exp ablation --mode debug
```

For a fuller run:

```bash
python Run_experiments.py --exp all --mode full
```

## Example Experiments

- baseline phase-field PINN for a notched specimen
- X-RAS-PINN with domain decomposition and interface loss
- ablation without interface loss
- ablation without residual adaptive sampling
- comparison between baseline and X-RAS midline predictions

## Skills Demonstrated

- scientific machine learning
- physics-informed neural networks
- phase-field fracture modelling
- domain decomposition
- adaptive sampling
- PyTorch experimentation
- model evaluation and ablation design

## Status

Research prototype. The repository is intended to document exploratory experiments rather than provide a polished simulation package.
