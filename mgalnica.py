import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def well_plotter(y_true, y_fit, time, norm_react_rates, well_names):

    # Simplified regular expression to match letter and number
    pattern = r'([A-H])(\d+)'

    # Find all matches using regular expressions
    matches = [re.match(pattern, item) for item in well_names]

    # Extract letters and numbers from the matches
    letters = [match.group(1) for match in matches]
    numbers = [int(match.group(2)) for match in matches]

    # Get unique letters as a list
    unique_letters = sorted(set(letters))

    rows, cols = len(unique_letters), numbers

    slope = norm_react_rates


    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.reshape((rows, cols))

    for i, letter in enumerate(letters):
        for j in range(1, cols + 1):
            ax = axes[i, j - 1]
            col_name = f"{letter}{j}"
        
            ax.plot(time, y_true[col_name], time, y_fit[col_name])

            # Name of well
            ax.text(0.1, 0.9, col_name, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

            # Value per well
            ax.text(0.1, 0.70, slope[j][i], transform=ax.transAxes,
            fontsize=10, va='top', ha='left', color = 'red')

            ax.tick_params(labelsize=6)

    fig.suptitle("Yield over Time for all wells")
    fig.supylabel("Yield")
    fig.supxlabel("Time(s)")
    fig.legend("Actual Data", loc = 'center', bbox_to_anchor = (1000, 0.5))

    fig.text(0.84, 0.96, "Actual data", fontweight = 'bold', color = '#FF7F0E')
    fig.text(0.84, 0.98, "Fit", fontweight = 'bold', color = '#257BB6')
    fig.text(0.795, 0.97, "Legend:", color = 'black')
    fig.text(0.91, 0.97, "Reaction Rate", fontweight = 'bold', color = 'red')

    plt.tight_layout()
    plt.show()


