import matplotlib.pyplot as plt
import numpy as np
import openmm.unit as unit
from experiments.utils import get_fig_axs, get_fig_ax, Colors, postprocess_style, setup_latex
from semi_md.utilities.md_plotting_helpers import parse_log_file, make_plots

if __name__ == '__main__':
    setup_latex()
    plot_store = np.load("artifacts/plot_store_openmm_protein_0.1 fs.npz")
    h, u = str(plot_store["timestep"]).split(" ")
    steps, potenergies, kinenergies, totenergies, temperatures = parse_log_file("artifacts/openmm_protein_0.1 fs.log")
    if u == "fs":
        steps = float(h) * np.array(steps)
    else:
        raise RuntimeError("Manually adjust step size")
    make_plots(steps, potenergies, kinenergies, totenergies, temperatures, f"figures/openmm_protein", factor=1)
