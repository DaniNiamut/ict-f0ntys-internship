import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

df1 = pd.read_excel("internship_simplified.xlsx")

time = df1['Time'].apply(lambda t: t.hour * 3600 + t.minute * 60 + t.second).to_numpy()

rows, cols = 8, 12
letters = list(string.ascii_uppercase[:rows])  # ['A', 'B', ..., 'H']
slope = np.round(np.random.uniform(0.05, 4.25, size=(93, 12)), 2)

fig, axes = plt.subplots(rows, cols, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.reshape((rows, cols))

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Generate column names: A1–A12, B1–B12, ..., H1–H12
columns = [f"{row}{col}" for row in "ABCDEFGH" for col in range(1, 13)]

# Simulate data: linear trend + noise
data = {
    col: (
        np.random.uniform(0.05, 0.1) +            # baseline
        np.random.uniform(1e-5, 5e-5) * time      # slope
    )
    for col in columns
}

# Build DataFrame
df = pd.DataFrame(data)
df.insert(0, "Time", time)

#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

for i, letter in enumerate(letters):
    for j in range(1, cols + 1):
        ax = axes[i, j - 1]
        col_name = f"{letter}{j}"
        
        if col_name in df.columns:
            ax.plot(time, df1[col_name], time, df[col_name])

            # Name of well
            ax.text(0.1, 0.9, col_name, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')

            # Value per well
            ax.text(0.1, 0.70, slope[j][i], transform=ax.transAxes,
            fontsize=10, va='top', ha='left', color = 'red')

        else:
            ax.set_visible(False)

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
