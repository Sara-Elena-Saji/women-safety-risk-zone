from flask import Flask, render_template, request
import pandas as pd
import folium
from folium.plugins import HeatMap
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():

    from flask import request

    # Get selected time
    selected_time = request.args.get("time", "All")

    # Load dataset
    df = pd.read_csv("data.csv")

    trend_summary = (
        df.groupby("Time_Period")["Risk_Score"]
        .mean()
        .reset_index()
    )

    most_risky_period = trend_summary.loc[
        trend_summary["Risk_Score"].idxmax()
    ]["Time_Period"]

    highest_avg_risk = round(
        trend_summary["Risk_Score"].max(), 2
    )


    # Filter by time
    if selected_time != "All":
        df = df[df["Time_Period"] == selected_time]

    # ---- Risk Classification ----
    df["Risk_Level"] = pd.cut(
        df["Risk_Score"],
        bins=[0, 0.4, 0.7, 1],
        labels=["Low", "Medium", "High"]
    )

    # ---- Count Risks ----
    low_count = (df["Risk_Level"] == "Low").sum()
    medium_count = (df["Risk_Level"] == "Medium").sum()
    high_count = (df["Risk_Level"] == "High").sum()

    # ---- Top 3 Risk Zones (Based on Filtered Data) ----
    top_zones = df.sort_values(by="Risk_Score", ascending=False).head(3)

    top_zones_list = top_zones[
        ["Latitude", "Longitude", "Risk_Score", "Time_Period"]
    ].to_dict(orient="records")

    # ---- Create Map ----
    m = folium.Map(
        location=[df["Latitude"].mean(), df["Longitude"].mean()],
        zoom_start=13
    )

    heat_data = [
        [row["Latitude"], row["Longitude"], row["Risk_Score"]]
        for _, row in df.iterrows()
    ]

    HeatMap(
        heat_data,
        gradient={
            0.2: "green",
            0.5: "yellow",
            0.8: "red"
        }
    ).add_to(m)

    # ---- Explainable Risk Markers ----
    for _, row in df.iterrows():
        popup_text = f"""
        <b>Location:</b> {row['Latitude']}, {row['Longitude']}<br>
        <b>Risk Score:</b> {row['Risk_Score']}<br>
        <b>Risk Level:</b> {row['Risk_Level']}<br>
        <b>Time Period:</b> {row['Time_Period']}
        """

        folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        color="pink",
        fill=True,
        fill_opacity=0.7,
        popup=popup_text
        ).add_to(m)

    os.makedirs("static", exist_ok=True)
    m.save("static/heatmap.html")

    return render_template(
    "dashboard.html",
    low_count=low_count,
    medium_count=medium_count,
    high_count=high_count,
    selected_time=selected_time,
    most_risky_period=most_risky_period,
    highest_avg_risk=highest_avg_risk,
    top_zones=top_zones_list
)


if __name__ == "__main__":
    app.run(debug=True)
