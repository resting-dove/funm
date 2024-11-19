import matplotlib.pyplot as plt


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


def make_plots(steps, potenergies, kinenergies, totenergies, temperatures, title=""):
    # Create figure and first axis
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()  # Get current axis
    ax2 = ax1.twinx()  # Create another axis that shares the same x-axis

    # Plot energy on the first y-axis
    ax1.plot(steps, potenergies, marker='o', linestyle='-', color='red', label='Energy')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Energy', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # Plot temperature on the second y-axis
    ax2.plot(steps, temperatures, marker='x', linestyle='-', color='blue', label='Temperature')
    ax2.set_ylabel('Temperature', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Title and grid
    plt.title(title + 'Energy and Temperature vs. Time Step')
    ax1.grid(True)

    plt.show()

    # Create figure and first axis
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()  # Get current axis
    ax2 = ax1.twinx()  # Create another axis that shares the same x-axis

    # Plot energy on the first y-axis
    ax1.plot(steps, kinenergies, marker='o', linestyle='-', color='red', label='Kinetic Energy')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Kinetic Energy', color='red')
    ax1.tick_params(axis='y', labelcolor='red')

    # Plot temperature on the second y-axis
    ax2.plot(steps, totenergies, marker='x', linestyle='-', color='blue', label='Total Energy')
    ax2.set_ylabel('Total Energy', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Title and grid
    plt.title(title + 'Energy and Temperature vs. Time Step')
    ax1.grid(True)

    plt.show()
