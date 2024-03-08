import os
from pathlib import Path
import ex1
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

TESTS_PATH = "./input"

def main():
    histograms1 = []
    histograms2 = []
    deltas1 = []
    deltas2 = []

    for obj in os.listdir(TESTS_PATH):
        p = Path(obj)
        name, extension = p.stem, p.suffix
        res = ex1.main(f"{TESTS_PATH}/{p}", extension)
        print(f"{obj}'s scene cut is at frames {res}")

    if len(histograms1) != 2 or len(histograms2) != 2:
        return
    
    histogram_graph(1, 1, histograms1[0][99], histograms2[0][100])
    histogram_graph(1, 2, histograms1[1][149], histograms2[1][150])
    histogram_graph(2, 1, histograms2[0][174], histograms2[0][175])
    histogram_graph(2, 2, histograms2[1][74], histograms2[1][75])
    delta_graph(1, deltas1)
    delta_graph(2, deltas2)

def histogram_graph(num, num2, histogram1, histogram2):
    df = pd.DataFrame({
        "frame": ["After Cut"] * len(histogram1) + ["Before Cut"] * len(histogram2),
        "gray-level": [i for i in range(len(histogram1))] * 2,
        "count": np.concatenate((histogram2, histogram1))
    })

    print(df)

    fig = px.bar(df, x="gray-level", y="count", color="frame", opacity=0.8,
        title=f"Histogram over frames (Category {num}, Video {num2})",
        labels={
            "x": "Gray Level",
            "y": "Count"
        },
    )

    fig.write_image(f"category{num}{num2}_histograms.png")

def delta_graph(num, deltas):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[i for i in range(len(deltas[0]))],
        y=deltas[0],
        name="Video 1"
    ))

    fig.add_trace(go.Scatter(
        x=[i for i in range(len(deltas[1]))],
        y=deltas[1],
        name="Video 2"
    ))

    fig.update_layout(
        title=f"Delta over frames (Category {num})",
        xaxis_title="Frame",
        yaxis_title="Delta",
        legend_title="Video Number"
    )

    fig.write_image(f"category{num}_deltas.png")

if __name__ == '__main__':
    main()
