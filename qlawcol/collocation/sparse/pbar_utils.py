import time
from contextlib import contextmanager
from threading import Thread

from tqdm import tqdm


@contextmanager
def ipopt_pbar_from_file(
    max_iter: int, fname: str = "IPOPT.out", desc: str = "IPOPT Progress"
):
    """
    Creates a progress bar which updates based on IPOPT output to a file.
    The IPOPT file format looks like
    ...
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
    1160  1.6635924e+03 5.75e-07 2.38e+01  -7.0 2.69e-01    -  1.00e+00 8.14e-02f  1
    1161  1.6635549e+03 4.16e-06 2.35e+01  -7.1 5.06e+00    -  2.41e-02 1.42e-02f  1
    ...
    iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
    ###
    ###

    - we can extract the iteration number and objective value from each line, and update the progress bar accordingly.
    - the progress bar needs to run on a different thread than the IPOPT optimization, so we can use a context manager to handle the lifecycle of the progress bar.
    """

    pbar = tqdm(desc=desc, total=max_iter)
    stop_thread = False

    # create a dummy file if it doesn't exist, and clear it if it does exist
    with open(fname, "w") as f:
        pass

    def update_pbar():
        with open(fname, "r") as f:
            last_line = ""
            while not stop_thread:
                line = f.readline()
                if not line:
                    time.sleep(0.1)  # Sleep briefly to avoid busy-waiting
                    continue
                last_line = line

                if last_line.startswith("iter"):
                    continue
                parts = last_line.split()
                if len(parts) < 3:
                    continue
                try:
                    iter_num = int(parts[0])
                    obj_val = float(parts[1])
                    constr_violation = float(parts[2])
                    steptype = parts[-2][-1]  # last character of second to last column
                    pbar.n = iter_num
                    pbar.set_postfix(
                        {
                            "obj": f"{obj_val:.4e}",
                            "cstr": f"{constr_violation:.4e}",
                            "step": steptype,
                        }
                    )
                    pbar.refresh()
                except (ValueError, IndexError):
                    continue

    thread = Thread(target=update_pbar)
    thread.start()

    try:
        yield
    finally:
        stop_thread = True
        thread.join()
        pbar.close()
