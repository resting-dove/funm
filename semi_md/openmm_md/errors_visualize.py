import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mdtraj
import openmm.app as app
import openmm.unit as unit
from semi_md.utilities.md_plotting_helpers import parse_log_file
from experiments.utils import get_fig_ax, Colors, postprocess_style, setup_latex

color_counter = 0


def gather_stuff(name):
    trajfilepath = "artifacts/" + name + "_trajectory.dcd"
    trajectory: mdtraj.Trajectory = mdtraj.load(
        trajfilepath, top=mdtraj.Topology.from_openmm(topopoly)
    )
    logfilepath = "artifacts/openmm_protein_" + name + ".log"
    steps, potenergies, kinenergies, totenergies, temperatures = parse_log_file(logfilepath)
    plotstorepath = "artifacts/plot_store_openmm_protein_" + name + ".npz"
    plot_store = np.load(plotstorepath)
    time_step = float(str(plot_store["timestep"]).split(" ")[0]) * unit.femtosecond
    tsteps = (np.array(steps) * time_step).value_in_unit(unit.femtosecond)
    return trajectory, steps, potenergies, kinenergies, totenergies, plot_store, time_step, tsteps


def plot(trajectory, rtrajectory, tsteps, ax, label):
    maxrele = []
    for i in range(len(steps)):
        pos = np.array(trajectory.openmm_positions(i)._value)
        rpos = np.array(rtrajectory.openmm_positions(i)._value)
        # relerrors = np.sum(np.abs(pos - rpos) / np.abs(pos), axis=1)
        relerrors = np.linalg.norm(pos - rpos, axis=1) / np.linalg.norm(pos, axis=1)
        maxrele.append(max(relerrors))
    ax.plot(tsteps, maxrele, label=label, color=color[color_counter])


def plot_energy(totenergies, rtotenergies, tsteps, ax2, label):
    global color_counter  # sorry
    ax2.plot(tsteps, totenergies, linestyle='-', label=label, color=color.get(color_counter))
    ax2.plot(tsteps, rtotenergies, linestyle=':', color=color.get(color_counter))
    color_counter += 1


if __name__ == '__main__':
    setup_latex()
    color = Colors()
    topopoly = app.PDBFile("../preparation/minimized_structure.pdb").getTopology()
    name = "OSGS99_0.05fs_LaWkm_80"
    trajectory, steps, potenergies, kinenergies, totenergies, plot_store, time_step, tsteps = gather_stuff(name)

    name = "OSGS99_0.05fs_ALaWkm_4"
    rtrajectory, rsteps, rpotenergies, rkinenergies, rtotenergies, rplot_store, rtime_step, rtsteps = gather_stuff(name)

    fig2, ax2 = get_fig_ax(ratio=2)
    fig, ax = get_fig_ax()
    plot(trajectory, rtrajectory, tsteps, ax, "0.05 fs")
    plot_energy(totenergies, rtotenergies, tsteps, ax2, "0.05 fs")

    # No .npz file bc blow up
    #############################################
    name = "OSGS99_0.1fs_LaWkm_80"
    # trajectory, steps, potenergies, kinenergies, totenergies, plot_store, time_step, tsteps = gather_stuff(name)
    #
    trajfilepath = "artifacts/" + name + "_trajectory.dcd"
    trajectory: mdtraj.Trajectory = mdtraj.load(
        trajfilepath, top=mdtraj.Topology.from_openmm(topopoly)
    )
    logfilepath = "artifacts/openmm_protein_" + name + ".log"
    steps, potenergies, kinenergies, totenergies, temperatures = parse_log_file(logfilepath)
    name = "OSGS99_0.1fs_ALaWkm_4"
    # rtrajectory, rsteps, rpotenergies, rkinenergies, rtotenergies, rplot_store, rtime_step, rtsteps = gather_stuff(name)
    # plot(trajectory, rtrajectory, tsteps, "0.1 fs")
    trajfilepath = "artifacts/" + name + "_trajectory.dcd"
    rtrajectory: mdtraj.Trajectory = mdtraj.load(
        trajfilepath, top=mdtraj.Topology.from_openmm(topopoly)
    )
    logfilepath = "artifacts/openmm_protein_" + name + ".log"
    rsteps, rpotenergies, rkinenergies, rtotenergies, rtemperatures = parse_log_file(logfilepath)

    maxrele = []
    for i in range(len(rsteps)):
        pos = np.array(trajectory.openmm_positions(i)._value)
        rpos = np.array(rtrajectory.openmm_positions(i)._value)
        # relerrors = np.sum(np.abs(pos - rpos) / np.abs(pos), axis=1)
        relerrors = np.linalg.norm(pos - rpos, axis=1) / np.linalg.norm(pos, axis=1)
        maxrele.append(max(relerrors))
    ax.plot(tsteps, maxrele, label="0.1 fs", color=color.get(color_counter))
    plot_energy(totenergies[:len(rsteps)], rtotenergies, tsteps, ax2, "0.1 fs")
    ######################################################

    name = "OSGS99_0.5fs_LaWkm_80"
    trajectory, steps, potenergies, kinenergies, totenergies, plot_store, time_step, tsteps = gather_stuff(name)

    name = "OSGS99_0.5fs_ALaWkm_4"
    rtrajectory, rsteps, rpotenergies, rkinenergies, rtotenergies, rplot_store, rtime_step, rtsteps = gather_stuff(name)
    plot(trajectory, rtrajectory, tsteps, ax, "0.5 fs")
    plot_energy(totenergies, rtotenergies, tsteps, ax2, "0.5 fs")

    name = "OSGS99_1fs_LaWkm_80"
    trajectory, steps, potenergies, kinenergies, totenergies, plot_store, time_step, tsteps = gather_stuff(name)

    name = "OSGS99_1fs_ALaWkm_4"
    rtrajectory, rsteps, rpotenergies, rkinenergies, rtotenergies, rplot_store, rtime_step, rtsteps = gather_stuff(name)
    plot(trajectory, rtrajectory, tsteps, ax, "1 fs")
    plot_energy(totenergies, rtotenergies, tsteps, ax2, "1 fs")

    # # No .npz file bc blow up
    # ############################################
    # name = "OSGS99_2fs_LaWkm_80"
    # # trajectory, steps, potenergies, kinenergies, totenergies, plot_store, time_step, tsteps = gather_stuff(name)
    # #
    # trajfilepath = "artifacts/" + name + "_trajectory.dcd"
    # trajectory: mdtraj.Trajectory = mdtraj.load(
    #     trajfilepath, top=mdtraj.Topology.from_openmm(topopoly)
    # )
    # logfilepath = "artifacts/openmm_protein_" + name + ".log"
    # steps, potenergies, kinenergies, totenergies, temperatures = parse_log_file(logfilepath)
    # name = "OSGS99_2fs_ALaWkm_4"
    # # rtrajectory, rsteps, rpotenergies, rkinenergies, rtotenergies, rplot_store, rtime_step, rtsteps = gather_stuff(name)
    # # plot(trajectory, rtrajectory, tsteps, "0.1 fs")
    # trajfilepath = "artifacts/" + name + "_trajectory.dcd"
    # rtrajectory: mdtraj.Trajectory = mdtraj.load(
    #     trajfilepath, top=mdtraj.Topology.from_openmm(topopoly)
    # )
    # maxrele = []
    # for i in range(len(steps)):
    #     pos = np.array(trajectory.openmm_positions(i)._value)
    #     rpos = np.array(rtrajectory.openmm_positions(i)._value)
    #     # relerrors = np.sum(np.abs(pos - rpos) / np.abs(pos), axis=1)
    #     relerrors = np.linalg.norm(pos - rpos, axis=1) / np.linalg.norm(pos, axis=1)
    #     maxrele.append(max(relerrors))
    # ax.plot(tsteps[:len(steps)], maxrele, label="2 fs")
    # ####################################################

    ax.set_yscale("log")
    # ax.set_ylim(bottom=np.finfo(np.float64).eps / 100)
    ax.set_xlabel('t [fs]')
    ax.set_ylabel('max rel. error')
    ax.legend()
    postprocess_style()
    fig.tight_layout()
    fig.savefig("figures/errors_visualize")

    ax2.set_ylim(top=5 * 1e5)
    ax2.set_xlabel('t [fs]')
    ax2.set_ylabel('Total energy [kJ/mol]')
    ax2.legend()
    postprocess_style()
    sns.despine(ax=ax2)
    fig2.tight_layout()
    fig2.savefig("figures/total_energy")
    1 + 1
