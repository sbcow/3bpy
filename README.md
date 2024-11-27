# 3bpy

This repository contains code to analyse the CR3BP. The library is written in Python, with a branch
in Julia for Symbolic Regression experiments.

The main 'executable' file is test.py / test.ipynb, which executes most functionalities in the library.

## Usage requirements

A environment.yaml file is attached that can be used to get all the dependencies.

## Main functionalities

- Analyse and use the dynamics of the CR3BP symbolically (using sympy) or numerically.
- Obtain initial guesses for periodic orbits (mainly Lyapunov) using the Monodromy matrix or the
third order Richardson approximation.
- Perform differential correction to close a given initial guess using Newton's method.
- Perform natural parameter or pseudo-arc continuation on a given periodic orbit to obtain a family
  of orbits.




