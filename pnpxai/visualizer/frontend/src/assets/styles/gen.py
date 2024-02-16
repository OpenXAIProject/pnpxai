# Code for generating RGB colors from a given colormap in Matplotlib.

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import json

def generate_rgb_colors(colormap_name, num_colors):
    """
    Generates RGB colors from a given colormap in Matplotlib.

    Args:
    - colormap_name (str): The name of the colormap (e.g., 'viridis', 'jet').
    - num_colors (int): The number of RGB colors to generate.

    Returns:
    - List of RGB colors.
    """
    # Get the colormap
    colormap = cm.get_cmap(colormap_name, num_colors)
    
    # Generate a list of values from 0 to 1 evenly spaced
    values = np.linspace(0, 1, num_colors)
    rescaled = 2 * (values - 0.5)
    
    # Generate RGB colors
    rgb_colors = []
    for idx, val in enumerate(values):
        r, g, b = colormap(val)[:3]
        r, g, b = int(r*255), int(g*255), int(b*255)
        rgb_colors.append(
            [val.round(3), f"rgb({r}, {g}, {b})"]
        )
        # rgb_colors.append(
        #     [rescaled[idx].round(3), f"rgb({r}, {g}, {b})"]
        # )
    return rgb_colors

# Example usage
colormap_name = 'bwr'
num_colors = 9
rgb_colors = generate_rgb_colors(colormap_name, num_colors)

print("RGB Colors (scaled 0-255):")
print(json.dumps(rgb_colors))
