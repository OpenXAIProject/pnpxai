import plotly.graph_objs as go
from plotly.subplots import make_subplots
from PIL import Image
import numpy as np
import random

# Load your image
img = Image.open("heatmap.png")
width, height = img.size

# Increase size for better visibility in subplot
width = width * 3.5
height = height * 2

# Convert image to array
img_array = np.array(img)

# Create a trace for the image
image_trace = go.Image(z=img_array)

# Generate random scores for infidelity and sensitivity
infidelity_score = random.uniform(0, 100)
sensitivity_score = random.uniform(0, 100)

# Create a trace for the evaluation score chart
score_chart_trace = go.Bar(
    x=['Infidelity', 'Sensitivity'],
    y=[infidelity_score, sensitivity_score],
    marker=dict(color=["blue", "green"])
)

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.5, 0.5],
    subplot_titles=("Image", "Evaluation Score Chart")
)

# Add image trace to the first subplot
fig.add_trace(image_trace, row=1, col=1)

# Add bar chart trace to the second subplot
fig.add_trace(score_chart_trace, row=1, col=2)

# Update layout
fig.update_layout(
    title="Image and Evaluation Score Chart",
    width=width, 
    height=height
)

# Save the figure as an HTML file
fig.write_html("image_and_score_plot.html")
