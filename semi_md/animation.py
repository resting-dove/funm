import numpy as np
from matplotlib import animation
from IPython.display import HTML
import matplotlib.pyplot as plt


def animate_md(r: np.array, topology, step=100, return_as="HTML"):
    # Setup the figure and axes...
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set(xlim=(-.35, .35), ylim=(-.35, .35), ylabel='Angstrom',
           xlabel='Angstrom', title=f'{next(topology.molecules).name} movement')

    ## drawing and animating
    scat = ax.scatter(r[0, :, 0], r[0, :, 1], marker='o', c=range(topology.n_atoms), s=500)

    def animate(i):
        scat.set_offsets(r[step * i])

    ani = animation.FuncAnimation(fig, animate, frames=int(len(r) / step))
    if return_as == "HTML":
        return HTML(ani.to_jshtml())
    else:
        return ani
