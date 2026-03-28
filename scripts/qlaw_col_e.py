import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from qlawcol.driver import ProblemData, Trajectory, optimize_transfer
from qlawcol.postprocess import plot_results
from qlawcol.qlaw.control import QLawParams

problem_data = ProblemData(
    initial_kep=np.array([24505.9e3, 0.725, np.radians(0.06), 0, 0, 0]),
    initial_mass=2000.0,
    qlaw_params=QLawParams(
        target=np.array([26500e3, 0.7, np.radians(116), 0, 0]),
        w_oe=np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
        eta=0.1,
        w_pen=1,
    ),
    t_max=200 * 86400,
    thrust=2,
    exhaust_velocity=2000 * 9.81,
    ode_maxsteps=32768,
    col_segments_per_rev=20,
    qlaw_tol=6e-2,
)
casename = "case_e_j2"
# check if we passed in a folder with existing result. if so, bypass optimization and just load the results
if len(sys.argv) > 1:
    folder = Path(sys.argv[1])

    print(f"Loading results from folder: {folder}")
    col_sol = Trajectory.load_from_file(folder / f"col_sol_{casename}.npz")
    qlaw_sol = Trajectory.load_from_file(folder / f"qlaw_sol_{casename}.npz")

else:
    res = optimize_transfer(problem_data, max_iter=5000)

    # save solution to file
    col_sol = res.collocation
    qlaw_sol = res.qlaw
    col_sol.dump_to_file(f"col_sol_{casename}.npz")
    qlaw_sol.dump_to_file(f"qlaw_sol_{casename}.npz")
# report delta-v
dv_col = np.log(col_sol.mass[0] / col_sol.mass[-1]) * problem_data.exhaust_velocity
dv_q = np.log(qlaw_sol.mass[0] / qlaw_sol.mass[-1]) * problem_data.exhaust_velocity
print(f"Collocation Delta-V: {dv_col:.2f} m/s")
print(f"Q-law Delta-V: {dv_q:.2f} m/s")
print(f"Delta-V reduction: {(dv_q - dv_col) / dv_q * 100:.2f} %")

plot_results(col_sol, qlaw_sol, problem_data, plot_prefix=casename)

plt.show()
