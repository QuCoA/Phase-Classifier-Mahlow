import numpy as np
import hamiltonians as h
import correlators as c
import multiprocessing as mp
import itertools as it
from tqdm import tqdm
from timeit import default_timer as timer
from datetime import datetime
from sys import argv
import pandas as pd
import os


def generate_data(N, n_cores, to_generate=["H1", "H2", "H3"]):
    T = [-1.0, -1.0, -1.0]
    corr = c.Correlators(N)
    n_sup = int(N / 2 + 1)

    ############### PARAMETERS ###############

    Jzs, Ds = np.arange(-4, 4.1, 0.1, ), np.arange(-4, 4, 0.1)
    H1_params = it.product([N], Jzs, Ds, [corr], [n_sup])

    deltas = np.arange(0, 1.01, 0.0125)
    Deltas = np.arange(-1.5, 2.51, 0.05)
    H2_params = it.product([N], Deltas, deltas, [corr], [n_sup])

    thetas = pd.read_csv("data/thetas.csv", header=None).values.flatten()
    H3_params = it.product([N], thetas, [corr], [n_sup])

    ##########################################

    if "H1" in to_generate:
        lines = []
        start = timer()
        with mp.Pool(n_cores) as pool:
            lines = list(
                tqdm(
                    pool.imap(gen_H1, H1_params),
                    total=len(Jzs) * len(Ds),
                    desc="Calculating H1 correlations",
                )
            )

        with open(f"data/H1/N={N}.csv", "w") as fh:
            fh.write(
                "Jz, D, "
                + ", ".join(
                    [f"S1S{i}{j}" for i in range(1, n_sup + 1) for j in ["x", "y", "z"]]
                )
                + ", "
                + ", ".join([f"prodSi{j}" for j in ["x", "y", "z"]])
            )
            fh.writelines(lines)

        T[0] = timer() - start

    if "H2" in to_generate:
        lines = []
        start = timer()
        with mp.Pool(n_cores) as pool:
            lines = list(
                tqdm(
                    (pool.imap(gen_H2, H2_params)),
                    total=len(Deltas) * len(deltas),
                    desc="Calculating H2 correlations",
                )
            )
        with open(f"data/H2/N={N}.csv", "w") as fh:
            fh.write(
                "Delta, delta, "
                + ", ".join(
                    [f"S1S{i}{j}" for i in range(1, n_sup + 1) for j in ["x", "y", "z"]]
                )
                + ", "
                + ", ".join([f"prodSi{j}" for j in ["x", "y", "z"]])
            )
            fh.writelines(lines)

        T[1] = timer() - start

    if "H3" in to_generate:
        start = timer()
        lines = []
        with mp.Pool(n_cores) as pool:
            lines = list(
                tqdm(
                    pool.imap(gen_H3, H3_params),
                    total=len(thetas),
                    desc="Calculating H3 correlations",
                )
            )
            pass
        with open(f"data/H3/N={N}.csv", "w") as fh:
            fh.write(
                "theta, -1, "
                + ", ".join(
                    [f"S1S{i}{j}" for i in range(1, n_sup + 1) for j in ["x", "y", "z"]]
                )
                + ", "
                + ", ".join([f"prodSi{j}" for j in ["x", "y", "z"]])
            )
            fh.writelines(lines)

        T[2] = timer() - start

    return T


def gen_H1(args):
    N, Jz, D, corr, n_sup = args
    line = [Jz, D]
    try:
        H1 = h.XXZUniaxialSingleIonAnisotropy(N, Jz, D)
        gstate = H1.gstate
    except Exception:
        return ""

    for i in range(n_sup):
        line.append(np.real(np.vdot(gstate, corr.S1Six(i) @ gstate)))
        line.append(np.real(np.vdot(gstate, corr.S1Siy(i) @ gstate)))
        line.append(np.real(np.vdot(gstate, corr.S1Siz(i) @ gstate)))

    line.append(np.real(np.vdot(gstate, corr.prodSix @ gstate)))
    line.append(np.real(np.vdot(gstate, corr.prodSiy @ gstate)))
    line.append(np.real(np.vdot(gstate, corr.prodSiz @ gstate)))

    return "\n" + ", ".join([str(i) for i in line])


def gen_H2(args):
    N, Delta, delta, corr, n_sup = args
    line = [Delta, delta]
    try:
        H2 = h.BondAlternatingXXZ(N, Delta, delta)
        gstate = H2.gstate
    except Exception:
        return ""

    for i in range(n_sup):
        line.append(np.real(np.vdot(gstate, corr.S1Six(i) @ gstate)))
        line.append(np.real(np.vdot(gstate, corr.S1Siy(i) @ gstate)))
        line.append(np.real(np.vdot(gstate, corr.S1Siz(i) @ gstate)))

    line.append(np.real(np.vdot(gstate, corr.prodSix @ gstate)))
    line.append(np.real(np.vdot(gstate, corr.prodSiy @ gstate)))
    line.append(np.real(np.vdot(gstate, corr.prodSiz @ gstate)))
    return "\n" + ", ".join([str(i) for i in line])


def gen_H3(args):
    N, theta, corr, n_sup = args
    line = [theta, -1]

    try:
        H3 = h.BilinearBiquadratic(N, theta)
        gstate = H3.gstate
    except Exception:
        return ""

    for i in range(n_sup):
        line.append(np.real(np.vdot(gstate, corr.S1Six(i) @ gstate)))
        line.append(np.real(np.vdot(gstate, corr.S1Siy(i) @ gstate)))
        line.append(np.real(np.vdot(gstate, corr.S1Siz(i) @ gstate)))

    line.append(np.real(np.vdot(gstate, corr.prodSix @ gstate)))
    line.append(np.real(np.vdot(gstate, corr.prodSiy @ gstate)))
    line.append(np.real(np.vdot(gstate, corr.prodSiz @ gstate)))

    return "\n" + ", ".join([str(i) for i in line])


if __name__ == "__main__":
    N = int(argv[1]) if len(argv) > 1 else 8
    n_cores = int(argv[2]) if len(argv) > 2 else 2
    if n_cores > mp.cpu_count():
        n_cores = int(mp.cpu_count() / 2)
    to_generate = (
        argv[3:] if (len(argv) > 3) and (len(argv) < 7) else ["H1", "H2", "H3"]
    )

    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/H1"):
        os.makedirs("data/H1")
    if not os.path.exists("data/H2"):
        os.makedirs("data/H2")
    if not os.path.exists("data/H3"):
        os.makedirs("data/H3")
    if not os.path.exists("data/DataRuntime.dat"):
        with open("data/DataRuntime.dat", "w") as fh:
            fh.write("N, n_cores, H1, H2, H3, Date \n")

    print(f"""
        This program is set to calculate a {N}-chain spin-1
        correlation matrix for the following hamiltonians:
        {to_generate}.
        
        To do so, it is going to use {n_cores} of your CPUs.
        """)

    t1, t2, t3 = generate_data(N, n_cores, to_generate)

    with open("data/DataRuntime.dat", "a") as fh:
        fh.write(
            f"{N}, {n_cores} , {t1}, {t2}, {t3}, {datetime.today().strftime('%Y-%m-%d')} \n"
        )

    print(f"\nData successfuly generated.\nIt took: {t1 + t2 + t3} seconds")
