import matplotlib.pyplot as plt
import numpy as np

# 1. Sample Data
np.random.seed(42) # for reproducibility
x = np.random.rand(20)
y = np.random.rand(20)
num_points = len(x)

# 2. Define the index of the point to highlight
highlight_index = 5
highlight_color = 'red'
default_color = 'blue'

# 3. Create a list of colors (default color for all)
colors = [default_color] * num_points

# 4. Change the color for the specific point
if 0 <= highlight_index < num_points:
    colors[highlight_index] = highlight_color
else:
    print(f"Warning: Index {highlight_index} is out of bounds.")

# 5. Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=colors, s=50) # s=50 adjusts point size

plt.title(f'Scatter Plot Highlighting Point at Index {highlight_index} (Method 1)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()