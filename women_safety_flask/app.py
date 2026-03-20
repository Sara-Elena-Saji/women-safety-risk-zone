from flask import Flask, render_template, request, jsonify
import pandas as pd
import folium
from folium.plugins import HeatMap
import pickle
import os

app = Flask(__name__)

# Load trained model and encoders
model = pickle.load(open("risk_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():

    # Get selected time
    selected_time = request.args.get("time", "All")

    # Load dataset
    df = pd.read_csv("data.csv")

    # ---------------- ENCODE AND PREDICT RISK SCORE ----------------

    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    X = df[["Crime_Count", "Street_Light", "CCTV", "Police_Patrol", "Isolation_Level", "Time_Period"]]
    df["Risk_Score"] = model.predict(X)

    # Decode Time_Period back to original labels for filtering and display
    df["Time_Period"] = encoders["Time_Period"].inverse_transform(df["Time_Period"])

    # ---------------- TREND ANALYSIS ----------------

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

    # ---------------- FILTER BY TIME ----------------

    if selected_time != "All":
        df = df[df["Time_Period"] == selected_time]

    # ---------------- RISK CLASSIFICATION ----------------

    df["Risk_Level"] = pd.cut(
        df["Risk_Score"],
        bins=[0, 0.4, 0.7, 1],
        labels=["Low", "Medium", "High"]
    )

    # ---------------- COUNT RISKS ----------------

    low_count = (df["Risk_Level"] == "Low").sum()
    medium_count = (df["Risk_Level"] == "Medium").sum()
    high_count = (df["Risk_Level"] == "High").sum()

    # ---------------- TOP RISK ZONES ----------------

    top_zones = df.sort_values(by="Risk_Score", ascending=False).head(3)

    top_zones_list = top_zones[
        ["Latitude", "Longitude", "Risk_Score", "Time_Period"]
    ].to_dict(orient="records")

    # ---------------- CREATE MAP ----------------

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

    # ---------------- EXPLAINABLE MARKERS ----------------

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


@app.route("/add-data", methods=["POST"])
def add_data():
    data = request.get_json()

    new_row = {
        "Latitude": float(data["Latitude"]),
        "Longitude": float(data["Longitude"]),
        "Area_Name": data["Area_Name"],
        "Time_Period": data["Time_Period"],
        "Crime_Count": int(data["Crime_Count"]),
        "Street_Light": data["Street_Light"],
        "CCTV": data["CCTV"],
        "Police_Patrol": data["Police_Patrol"],
        "Isolation_Level": data["Isolation_Level"]
    }

    df = pd.read_csv("data.csv")
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("data.csv", index=False)

    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(debug=True)