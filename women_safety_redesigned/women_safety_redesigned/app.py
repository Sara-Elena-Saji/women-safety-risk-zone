from flask import Flask, render_template, request, jsonify
import pandas as pd
import folium
from folium.plugins import HeatMap
import pickle
import os
import json

app = Flask(__name__)

model    = pickle.load(open("risk_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

MIN_LAT = 10.480
MAX_LAT = 10.570
MIN_LON = 76.180
MAX_LON = 76.260


def get_risk_df():
    df = pd.read_csv("data.csv")
    df = df[
        (df["Latitude"].between(MIN_LAT, MAX_LAT)) &
        (df["Longitude"].between(MIN_LON, MAX_LON))
    ]
    encoded_df = df.copy()
    for col, le in encoders.items():
        encoded_df[col] = le.transform(encoded_df[col])
    X = encoded_df[["Crime_Count", "Street_Light", "CCTV",
                     "Police_Patrol", "Isolation_Level", "Time_Period"]]
    df = df.copy()
    df["Risk_Score"] = model.predict(X)
    df["Risk_Score"] = df["Risk_Score"].clip(0, 1)
    # decode Time_Period back
    df["Time_Period"] = encoders["Time_Period"].inverse_transform(encoded_df["Time_Period"])
    return df


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    selected_time = request.args.get("time", "All")

    df = pd.read_csv("data.csv")
    df = df[
        (df["Latitude"].between(MIN_LAT, MAX_LAT)) &
        (df["Longitude"].between(MIN_LON, MAX_LON))
    ]

    encoded_df = df.copy()
    for col, le in encoders.items():
        encoded_df[col] = le.transform(encoded_df[col])

    X = encoded_df[["Crime_Count", "Street_Light", "CCTV",
                     "Police_Patrol", "Isolation_Level", "Time_Period"]]
    df["Risk_Score"] = model.predict(X)
    df["Time_Period"] = encoders["Time_Period"].inverse_transform(
        encoded_df["Time_Period"]
    )

    trend_summary     = df.groupby("Time_Period")["Risk_Score"].mean().reset_index()
    most_risky_period = trend_summary.loc[trend_summary["Risk_Score"].idxmax()]["Time_Period"]
    highest_avg_risk  = round(trend_summary["Risk_Score"].max(), 2)

    if selected_time != "All":
        df = df[df["Time_Period"] == selected_time]

    df["Risk_Level"] = pd.cut(
        df["Risk_Score"], bins=[0, 0.4, 0.7, 1], labels=["Low", "Medium", "High"]
    )

    low_count    = (df["Risk_Level"] == "Low").sum()
    medium_count = (df["Risk_Level"] == "Medium").sum()
    high_count   = (df["Risk_Level"] == "High").sum()

    top_zones      = df.sort_values("Risk_Score", ascending=False).head(3)
    top_zones_list = top_zones[
        ["Latitude", "Longitude", "Risk_Score", "Time_Period"]
    ].to_dict(orient="records")

    m = folium.Map(location=[10.5276, 76.2144], zoom_start=14, tiles="OpenStreetMap")

    heat_data = [
        [float(r["Latitude"]), float(r["Longitude"]), float(r["Risk_Score"])]
        for _, r in df.iterrows()
    ]

    HeatMap(heat_data, radius=35, blur=28, min_opacity=0.4, max_zoom=18,
            gradient={0.0:"blue", 0.3:"green", 0.55:"yellow", 0.75:"orange", 1.0:"red"}
    ).add_to(m)

    for _, row in df.iterrows():
        color = {"High": "red", "Medium": "orange", "Low": "green"}.get(row["Risk_Level"], "green")
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=(f"<b>Area:</b> {row['Area_Name']}<br>"
                   f"<b>Risk Score:</b> {round(row['Risk_Score'],2)}<br>"
                   f"<b>Risk Level:</b> {row['Risk_Level']}<br>"
                   f"<b>Time Period:</b> {row['Time_Period']}")
        ).add_to(m)

    os.makedirs("static", exist_ok=True)
    m.save("static/heatmap.html")

    return render_template("dashboard.html",
        low_count=low_count, medium_count=medium_count, high_count=high_count,
        selected_time=selected_time, most_risky_period=most_risky_period,
        highest_avg_risk=highest_avg_risk, top_zones=top_zones_list)


@app.route("/add-data", methods=["POST"])
def add_data():
    data = request.get_json()
    lat  = float(data["Latitude"])
    lon  = float(data["Longitude"])

    if not (MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON):
        return jsonify({"status": "error", "message": "Location outside Thrissur city area"})

    new_row = {
        "Latitude": lat, "Longitude": lon,
        "Area_Name": data["Area_Name"], "Time_Period": data["Time_Period"],
        "Crime_Count": int(data["Crime_Count"]), "Street_Light": data["Street_Light"],
        "CCTV": data["CCTV"], "Police_Patrol": data["Police_Patrol"],
        "Isolation_Level": data["Isolation_Level"]
    }

    df = pd.read_csv("data.csv")
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("data.csv", index=False)
    return jsonify({"status": "success"})


@app.route("/safe-route")
def safe_route():
    df = get_risk_df()
    risk_zones = df[[
        "Latitude", "Longitude", "Area_Name", "Risk_Score",
        "Crime_Count", "Street_Light", "CCTV", "Police_Patrol",
        "Isolation_Level", "Time_Period"
    ]].to_dict(orient="records")
    return render_template("safe_route.html", risk_zones=json.dumps(risk_zones))


# ── Explainable AI ────────────────────────────────────────────────────────────
@app.route("/explain-route", methods=["POST"])
def explain_route():
    data  = request.get_json()
    zones = data.get("zones", [])

    if not zones:
        return jsonify({
            "summary": "No risk zones were found near this route.",
            "reasons": [],
            "zone_details": []
        })

    df = get_risk_df()

    names   = list({ z["name"] for z in zones })
    nearby  = df[df["Area_Name"].isin(names)].copy()

    if nearby.empty:
        return jsonify({
            "summary": "Could not retrieve feature data for zones near this route.",
            "reasons": [],
            "zone_details": []
        })

    # ✅ LIMIT zones (important fix)
    nearby = nearby.sort_values("Risk_Score", ascending=False).head(50)

    # ── Zone details ────────────────────────────────────────
    zone_details = []
    for _, row in nearby.head(5).iterrows():
        factors = []

        if row["Crime_Count"] >= 8:
            factors.append(f"Very high crime ({int(row['Crime_Count'])} incidents)")
        elif row["Crime_Count"] >= 5:
            factors.append(f"High crime ({int(row['Crime_Count'])} incidents)")
        elif row["Crime_Count"] >= 3:
            factors.append(f"Moderate crime ({int(row['Crime_Count'])} incidents)")

        if row["Street_Light"] == "Poor":
            factors.append("Poor street lighting")
        elif row["Street_Light"] == "Moderate":
            factors.append("Moderate street lighting")

        if row["CCTV"] == "No":
            factors.append("No CCTV coverage")

        if row["Police_Patrol"] == "Rare":
            factors.append("Rare police patrol")
        elif row["Police_Patrol"] == "Occasional":
            factors.append("Occasional police patrol")

        if row["Isolation_Level"] == "High":
            factors.append("Highly isolated area")
        elif row["Isolation_Level"] == "Medium":
            factors.append("Moderately isolated")

        if row["Time_Period"] == "Night":
            factors.append("Especially dangerous at Night")
        elif row["Time_Period"] == "Evening":
            factors.append("Elevated risk in Evening")

        level = "High Risk" if row["Risk_Score"] >= 0.7 else "Medium Risk"

        zone_details.append({
            "name": row["Area_Name"],
            "score": round(float(row["Risk_Score"]), 2),
            "level": level,
            "factors": factors
        })

    # ── Explainable AI (UPDATED) ───────────────────────────
    reasons = []

    max_crime = nearby["Crime_Count"].max()
    if max_crime >= 8:
        reasons.append("🔴 This route passes through areas with very high crime activity.")
    elif max_crime >= 5:
        reasons.append("🟠 Elevated crime levels detected along this route.")

    poor_count = int((nearby["Street_Light"] == "Poor").sum())
    if poor_count >= 3:
        reasons.append("🔴 Large parts of this route are poorly lit, reducing visibility and safety.")
    elif poor_count > 0:
        reasons.append("🟠 Some stretches of this route have poor lighting.")

    no_cctv_count = int((nearby["CCTV"] == "No").sum())
    if no_cctv_count >= 3:
        reasons.append("🔴 Many areas along this route lack CCTV surveillance.")
    elif no_cctv_count > 0:
        reasons.append("🟠 Limited CCTV coverage in some areas.")

    rare_count = int((nearby["Police_Patrol"] == "Rare").sum())
    if rare_count >= 3:
        reasons.append("🔴 Police presence is low in several parts of this route.")
    elif rare_count > 0:
        reasons.append("🟠 Some areas have limited police patrol.")

    iso_count = int((nearby["Isolation_Level"] == "High").sum())
    if iso_count >= 3:
        reasons.append("🔴 Several segments are isolated, making them less safe.")
    elif iso_count > 0:
        reasons.append("🟠 A few isolated stretches detected.")

    night_count = int((nearby["Time_Period"] == "Night").sum())
    if night_count >= 2:
        reasons.append("🔴 This route becomes significantly riskier at night.")

    if not reasons:
        reasons.append("🟡 This route has moderate risk with a mix of smaller safety concerns.")

    # ✅ Show only top 2 reasons
    reasons = reasons[:2]

    # ── Summary ─────────────────────────────────────────────
    high_count = int((nearby["Risk_Score"] >= 0.7).sum())
    med_count  = int(((nearby["Risk_Score"] >= 0.4) & (nearby["Risk_Score"] < 0.7)).sum())
    avg_score  = round(float(nearby["Risk_Score"].mean()), 2)

    summary = (
        f"This route passes through several risk-prone areas — "
        f"{high_count} high-risk and {med_count} medium-risk zones "
        f"(avg risk score: {avg_score})."
    )

    return jsonify({
        "summary": summary,
        "reasons": reasons,
        "zone_details": zone_details
    })
# ── Risk Chart data ───────────────────────────────────────────────────────────
@app.route("/risk-chart-data")
def risk_chart_data():
    try:
        df = get_risk_df()

        # 1. Risk distribution (SAFE)
        if "Risk_Level" not in df.columns:
            df["Risk_Level"] = pd.cut(
                df["Risk_Score"],
                bins=[0, 0.4, 0.7, 1.0],
                labels=["Low", "Medium", "High"]
            )

        dist = df["Risk_Level"].value_counts().reindex(
            ["Low", "Medium", "High"], fill_value=0
        ).to_dict()

        # 2. Time-based risk (ordered)
        time_order = ["Morning", "Afternoon", "Evening", "Night"]
        by_time = (
            df.groupby("Time_Period")["Risk_Score"]
            .mean()
            .round(3)
            .reindex(time_order)
            .dropna()
            .to_dict()
        )

        # 3. Top 10 risky areas
        top10 = (
            df.sort_values("Risk_Score", ascending=False)
            .head(10)[["Area_Name", "Risk_Score"]]
            .to_dict(orient="records")
        )

        # 4. Histogram (SAFE - no crash)
        import numpy as np
        counts, edges = np.histogram(df["Risk_Score"], bins=10, range=(0, 1))
        histogram = [
            {"label": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "count": int(counts[i])}
            for i in range(len(counts))
        ]

        # 5. Risk by street light
        by_light = (
            df.groupby("Street_Light")["Risk_Score"]
            .mean()
            .round(3)
            .to_dict()
        )

        # 6. Risk by patrol
        by_patrol = (
            df.groupby("Police_Patrol")["Risk_Score"]
            .mean()
            .round(3)
            .to_dict()
        )

        # 7. Top factors (SAFE)
        top_factors = {
            "poor_lighting": int((df["Street_Light"] == "Poor").sum()),
            "no_cctv": int((df["CCTV"] == "No").sum()),
            "rare_patrol": int((df["Police_Patrol"] == "Rare").sum()),
            "high_isolation": int((df["Isolation_Level"] == "High").sum()),
        }

        return jsonify({
            "distribution": dist,
            "by_time": by_time,
            "top10": top10,
            "histogram": histogram,
            "by_light": by_light,
            "by_patrol": by_patrol,
            "top_factors": top_factors
        })

    except Exception as e:
        print("ERROR in /risk-chart-data:", e)  # 👈 shows error in terminal
        return jsonify({"error": "Failed to load data"}), 500
    
if __name__ == "__main__":
    app.run(debug=True)