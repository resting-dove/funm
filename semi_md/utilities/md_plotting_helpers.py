import matplotlib.pyplot as plt
from experiments.utils import get_fig_ax, Colors, postprocess_style


def parse_log_file(log_file_path):
    print("Parsing log file")
    steps = []
    potenergies = []
    kinenergies = []
    totnenergies = []
    temperatures = []  # List to store temperature values
    with open(log_file_path, 'r') as file:
        for line in file:
            # Skip header or lines without numeric data
            try:
                step, potenergy, kinenergy, totenergy, temp = line.strip().split(',')
                steps.append(int(step))
                potenergies.append(float(potenergy))
                kinenergies.append(float(kinenergy))
                totnenergies.append(float(totenergy))
                temperatures.append(float(temp))  # Parse temperature
            except ValueError:
                continue
    return steps, potenergies, kinenergies, totnenergies, temperatures


def make_plots(steps, potenergies, kinenergies, totenergies, temperatures, save_path="", factor=1):
    # Create figure and first axis
    fig, ax = get_fig_ax(factor=factor)
    colors = Colors()
    ax2 = ax.twinx()  # Create another axis that shares the same x-axis

    # Plot energy on the first y-axis
    ax.plot(steps, potenergies, linestyle='-', color=colors[0], label='Pot. Energy')
    ax.set_xlabel('t [fs]')
    ax.set_ylabel('[kJ/mol]')
    ax.tick_params(axis='y')

    ax2.plot(steps, temperatures, linestyle="-", color=colors.get(1, True), label='Temp.')
    ax2.set_ylabel('Temperature [K]', color=colors.get(1, True))
    ax2.tick_params(axis='y', labelcolor=colors[1])
    ax2.set_ylim(bottom=0)

    ax.plot(steps, kinenergies, linestyle='-', color=colors[2], label='Kin. Energy')
    ax.plot(steps, totenergies, linestyle='-', color=colors[3], label='Tot. Energy')

    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
              ncols=3, mode="expand", borderaxespad=0.)
    # ax.legend(framealpha=.5, scatterpoints=1, numpoints=1)
    postprocess_style()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()
