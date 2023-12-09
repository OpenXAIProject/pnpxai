import plotly.graph_objs as go
import base64
import json

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, 'rb') as file:
        image = file.read()
    return base64.b64encode(image).decode('utf-8')

def get_json(image_path):
    # Encode the image
    encoded_image = encode_image(image_path)

    # Create a Plotly figure
    fig = go.Figure()

    # Add the image to the figure
    fig.add_layout_image(
        dict(
            source='data:image/jpg;base64,' + encoded_image,
            xref="x",
            yref="y",
            x=0,
            y=3,
            sizex=3,
            sizey=3,
            sizing="stretch",
            opacity=1.0,
            layer="below")
    )

    # Set the layout for the figure
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),  # Remove x-axis
        yaxis=dict(showgrid=False, zeroline=False, visible=False),  # Remove y-axis
        xaxis_showgrid=False, 
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        xaxis_range=[0,3],
        yaxis_range=[0,3],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),  # Remove margins
        width=240, 
        height=200
    )

    return fig.to_json()

if __name__ == '__main__':
    base_data = [
        {
            "experiment_id": 1,
            "name": "Experiment 1",
            "model" : "Model 1",
            "modelDetected" : True,
            "modelStructure" : "Model Structure"*40,
            "algorithms": ["Algorithm 1", "Algorithm 2", "Algorithm 3"],
            "data" : [
                {"sample_id" : 1, "name" : "Image 1", "json" : ""},
                {"sample_id" : 2, "name" : "Image 2", "json" : ""},
                {"sample_id" : 3, "name" : "Image 3", "json" : ""},
            ]
        },
        {          
            "experiment_id": 2,
            "name": "Experiment 2",
            "model" : "Model 2",
            "modelDetected" : True,
            "modelStructure" : "Model Structure"*40,
            "algorithms": ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4"],
            "data" : [
                {"sample_id" : 4, "name" : "Image 4", "json" : ""},
                {"sample_id" : 5, "name" : "Image 5", "json" : ""},
                {"sample_id" : 6, "name" : "Image 6", "json" : ""},
            ]
        },
        {          
            "experiment_id": 3,
            "name": "Experiment 3",
            "model" : "Model 3",
            "modelDetected" : False,
            "modelStructure" : "",
            "algorithms": ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4"],
            "data" : [
                {"sample_id" : 7, "name" : "Image 7", "json" : ""},
                {"sample_id" : 8, "name" : "Image 8", "json" : ""},
                {"sample_id" : 9, "name" : "Image 9", "json" : ""},
            ]
        }
    ]

    for experiment in base_data:
        for image in experiment["data"]:
            idx = str(image["sample_id"]).zfill(4)
            path = f"../data/caltech256/001.ak47/001_{idx}.jpg"
            print(get_json(path))
            exit()
            # image["json"] = get_json(path)


    with open("plotly_test.json", "w") as f:
        json.dump(base_data, f, indent=4)



    predictions = [
        {
            "experiment_id" : 1,
            "algorithm" : "Algorithm 1",
            "sample_id" : 1, "name" : "Image 1", "json" : "", 
            "trueLabel" : "label1",
            "prediction" : 
                [
                    {"label" : "label1", "value" : 0.9},
                    {"label" : "label2", "value" : 0.1},
                    {"label" : "label3", "value" : 0.0},
                ]
        },
        # ...
    ]

    evalutions = [
        {
            "experiment_id" : 1,
            "algorithm" : "Algorithm 1",
            "sample_id" : 1, "name" : "Image 1", "json" : "", 
            "metric1" : 10,
            "metric2" : 20,
        },
        # ...
    ]

    # prediction = []
    # for exp in base_data:
    #     for algo in exp['algorithms']:
    #         for img in exp['data']:
    #             prediction.append({
    #                 "experiment_id" : exp['experiment_id'],
    #                 "algorithm" : algo,
    #                 "sample_id" : img['sample_id'],
    #                 "trueLabel" : "label1",
    #                 "prediction" : 
    #                     [
    #                         {"label" : "label1", "value" : 0.9},
    #                         {"label" : "label2", "value" : 0.1},
    #                         {"label" : "label3", "value" : 0.0},
    #                     ],
    #                 "isCorrect" : True
    #             })


import plotly.express as px
import json

# Sample data
df = px.data.iris()

# Create a simple scatter plot
fig = px.scatter(df, x='sepal_width', y='sepal_length', color='species')

# Convert the plot to JSON
plot_json = json.dumps(fig, cls=px.utils.PlotlyJSONEncoder)

# Now, plot_json contains the JSON representation of the plot
