import time
from contextlib import contextmanager
from threading import Thread

from tqdm import tqdm


@contextmanager
def ipopt_pbar_from_file(
    max_iter: int,
    fname: str = "IPOPT.out",
    desc: str = "IPOPT",
):
    """
    Creates a progress bar which updates based on IPOPT output to a file.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations for IPOPT (used to set the total for the progress bar).
    fname : str, optional
        The filename where IPOPT writes its output, by default "IPOPT.out".
    desc : str, optional
        Description for the progress bar, by default "IPOPT".
    Usage
    -----
    with ipopt_pbar_from_file(max_iter=1000):
        (run IPOPT optimization here)
    """

    pbar = tqdm(desc=desc, total=max_iter)
    stop_thread = False

    # create a dummy file if it doesn't exist, and clear it if it does exist
    with open(fname, "w") as _:
        pass

    def update_pbar():
        with open(fname, "r") as f:
            while not stop_thread:
                line = f.readline()
                if not line:
                    time.sleep(0.1)  # Sleep briefly to avoid busy-waiting
                    continue
                if line.startswith("iter"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                if not (parts[0].strip().isdigit() or parts[0].split("r")[0].isdigit()):
                    continue

                try:
                    in_restoration = "r" in parts[0]
                    if in_restoration:
                        iter_num = int(parts[0].split("r")[0])
                        obj = float(parts[0].split("r")[1])
                        constr_violation = float(parts[1])
                    else:
                        iter_num = int(parts[0])
                        obj = float(parts[1])
                        constr_violation = float(parts[2])
                    steptype = parts[-2][-1]  # last character of second to last column
                    pbar.n = iter_num
                    pbar.set_postfix(
                        {
                            "obj": f"{obj:.4e}",
                            "infeas": f"{constr_violation:.2e}",
                            "st": steptype,
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
