# Energy-optimal Low-thrust Many-revolution Elliptical Orbit Transfers Using Feedback Laws for an Initial Trajectory Guess

This repository contains the source code for my AER1503 project, "Energy-optimal Low-thrust Many-revolution Elliptical Orbit Transfers Using Feedback Laws for an Initial Trajectory Guess".

## Installation

### Python Package

After cloning the repository and creating [a virtual environment](https://docs.python.org/3/library/venv.html) run

```bash
pip install -e .
```

This will install the `qlawcol` package and its dependencies.

### IMPORTANT: Install IPOPT

IPOPT is required to run large-scale optimization. You can install IPOPT in several different ways depending on your system. One possible method is through your system's package manager on Linux, or using Conda. The installation process may be highly nontrivial, so it is recommended to consult the [IPOPT documentation](https://coin-or.github.io/Ipopt/INSTALL.html) for detailed instructions.

Furthermore, you can install higher-performance linear solvers such as HSL MA97 (free for academics). I used MUMPS without issue for my work, but the higher-end solvers may provide better performance for larger problems.

## Reproducing Case Studies

The numerical case studies from the paper can be reproduced by running the scripts in the `code/scripts` directory. The relevant scripts are `qlaw_col_a.py`, `qlaw_col_b.py`, `qlaw_col_c.py`, `qlaw_col_e.py`, and `qlaw_col_hlgl.py`.

These scripts do not directly map 1:1 with the cases in the paper, and some customization is required to reproduce specific cases. For example, including J2 perturbations requires modifying the code to add the `j2=True` argument to the constructor of the `ProblemData` class.

To run a case study, navigate to the `code` directory and execute the desired script. For example, to run Case A:

```bash
python scripts/qlaw_col_a.py
```

After the Q-law finishes running, a progress bar showing the status of IPOPT will appear. After finishing the optimization run, the trajectories will be saved to `.npz` files. Additionally, plots are produced.

## Miscellaneous Scripts

`scripts/petr_a1.py` and `scripts/petr_a2.py` are used to reproduce the Q-law solutions from Petropoulos' paper.

`orbit_transfer_2d.py` shows a trajectory optimization run WITHOUT have a Q-law initialization. This was not discussed in the report, but it is easy to find examples of transfers where the included ad-hoc initialization method fails to produce a good initial guess. `orbit_transfer_hlgl.py` is a similar script, swapping out HS collocation for HLGL collocation.

`col_demo.py` and `hlgl_example.py` are simple scripts to verify the correct implementation of the collocation methods. They solve a simple problem with a known solution and plot the results. `col_demo.py` uses HS collocation, while `hlgl_example.py` uses HLGL collocation.



