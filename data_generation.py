import numpy as np
import hamiltonians as h
import correlators as c
from tqdm import tqdm # for progress bar
from timeit import default_timer as timer
from datetime import datetime
from sys import argv

def generate_data(N, to_generate = ['H1', 'H2', 'H3']):

    t1, t2, t3 = -1, -1, -1

    ###########################################
    H_1_parameters = np.arange(-4, 4, 0.1)
    Jzs, Ds = H_1_parameters, H_1_parameters

    
    # H2 parameters
    # I'm pretty sure that the original article
    # is wrong here.

    deltas = np.arange(0, 1, 0.0125)
    Deltas = np.arange(-1.5,2.5, 0.05)

    # H3 parameters
    thetas = np.arange(0,2*np.pi,0.001*np.pi)

    ##########################################

    corr = c.Correlators(N)
    n_sup = int(N/2+1)

    if 'H1' in to_generate:
        start = timer()
        with open(f"data/H1/N={N}.csv", 'w') as fh:
            fh.write("Jz, D, " + ", ".join([f"S1S{i}{j}" for i in range(1, n_sup+1) for j in ['x', 'y', 'z']]) + ", "+
            ", ".join([f"prodSi{j}" for j in ['x', 'y', 'z']]))

            for Jz in tqdm(Jzs, desc="Solving H1 correlations"):
                for D in Ds:
                    line = [Jz, D]
                    try: 
                        H1 = h.XXZUniaxialSingleIonAnisotropy(N, Jz, D)
                        gstate = H1.gstate
                    except: 
                        continue  
                    gstate_dagg = np.conj(gstate).T
                    for i in range(n_sup):
                        line.append(np.real(gstate_dagg @ corr.S1Six(i) @ gstate))
                        line.append(np.real(gstate_dagg @ corr.S1Siy(i) @ gstate))
                        line.append(np.real(gstate_dagg @ corr.S1Siz(i) @ gstate))
                    
                    line.append(np.real(gstate_dagg @ corr.prodSix @ gstate))
                    line.append(np.real(gstate_dagg @ corr.prodSiy @ gstate))
                    line.append(np.real(gstate_dagg @ corr.prodSiz @ gstate))

                    fh.write("\n"+", ".join([str(i) for i in line]))
        t1 = timer() - start

    if 'H2' in to_generate:
        start = timer()
        with open(f"data/H2/N={N}.csv", 'w') as fh:
            fh.write("Delta, delta, " + ", ".join([f"S1S{i}{j}" for i in range(1, n_sup+1) for j in ['x', 'y', 'z']]) + ", "+
            ", ".join([f"prodSi{j}" for j in ['x', 'y', 'z']]))


            for Delta in tqdm(Deltas, desc="Solving H2 correlations"):
                for delta in deltas:
                    line = [Delta, delta]
                    try:
                        H2 = h.BondAlternatingXXZ(N, Delta, delta)
                        gstate = H2.gstate
                    except:
                        continue 
                    gstate_dagg = np.conj(gstate).T
                    for i in range(n_sup):
                        line.append(np.real(gstate_dagg @ corr.S1Six(i) @ gstate))
                        line.append(np.real(gstate_dagg @ corr.S1Siy(i) @ gstate))
                        line.append(np.real(gstate_dagg @ corr.S1Siz(i) @ gstate))
                    
                    line.append(np.real(gstate_dagg @ corr.prodSix @ gstate))
                    line.append(np.real(gstate_dagg @ corr.prodSiy @ gstate))
                    line.append(np.real(gstate_dagg @ corr.prodSiz @ gstate))

                    fh.write("\n"+", ".join([str(i) for i in line]))
        t2 = timer() - start
    
    if 'H3' in to_generate:
        start = timer()
        with open(f"data/H3/N={N}.csv", 'w') as fh:
            fh.write("theta, -1, " + ", ".join([f"S1S{i}{j}" for i in range(1, n_sup+1) for j in ['x', 'y', 'z']]) + ", "+
            ", ".join([f"prodSi{j}" for j in ['x', 'y', 'z']]))


            for theta in tqdm(thetas, desc="Solving H3 correlations"):
                line = [theta, -1]
                try:
                    H3 = h.BilinearBiquadratic(N, theta)
                    gstate = H3.gstate
                except:
                    continue
                gstate_dagg = np.conj(gstate).T
                for i in range(n_sup):
                    line.append(np.real(gstate_dagg @ corr.S1Six(i) @ gstate))
                    line.append(np.real(gstate_dagg @ corr.S1Siy(i) @ gstate))
                    line.append(np.real(gstate_dagg @ corr.S1Siz(i) @ gstate))
                    
                line.append(np.real(gstate_dagg @ corr.prodSix @ gstate))
                line.append(np.real(gstate_dagg @ corr.prodSiy @ gstate))
                line.append(np.real(gstate_dagg @ corr.prodSiz @ gstate))
            
                fh.write("\n"+", ".join([str(i) for i in line]))
        t3 = timer() - start

    return t1, t2, t3

N = int(argv[1]) if len(argv) > 1 else 8
to_generate = argv[2:] if (len(argv) > 2) and (len(argv) < 6) else ['H1', 'H2', 'H3']

t1, t2, t3 = generate_data(N, to_generate)
with open('data/DataRuntime.dat', 'a') as fh:
    fh.write(f"{N}, {t1}, {t2}, {t3}, {datetime.today().strftime('%Y-%m-%d')} \n")
    
print(f"\nData successfuly generated.\nIt took: {t1 + t2 + t3} seconds") 
