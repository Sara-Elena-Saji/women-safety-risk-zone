from flask import Flask, render_template, request
import pandas as pd
import folium
from folium.plugins import HeatMap
import os
import requests

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():

    # Get selected time
    selected_time = request.args.get("time", "All")

    # Load dataset
    df = pd.read_csv("data.csv")

    # ---------------- NEWS FEATURE ----------------

    API_KEY = "137f3df10f964630bb1de2387b9bec71"

    news_url = f"https://newsapi.org/v2/everything?q=((kochi OR kerala) AND (rape OR assault OR harassment OR stalking OR \"violence against women\"))&language=en&sortBy=publishedAt&apiKey={API_KEY}"    
    news_data = requests.get(news_url).json()

    articles = news_data.get("articles", [])[:3]

    news_list = []

    negative_words = [
    "assault","rape","harassment","attack","crime",
    "violence","molest","stalk","abuse","arrest","police"
]

    positive_words = [
    "empowerment","achievement","success","education",
    "award","leadership","initiative","program","scheme"
]

    for article in articles:

        title = (article["title"] or "").lower()
        sentiment = "neutral"

        if not any(word in title for word in negative_words + positive_words):
         continue

        if any(word in title for word in negative_words):
            sentiment = "negative"

        elif any(word in title for word in positive_words):
            sentiment = "positive"

    news_list.append({
        "title": article["title"],
        "url": article["url"],
        "sentiment": sentiment
    })

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
        top_zones=top_zones_list,
        news_list=news_list
    )


if __name__ == "__main__":
      app.run(host="0.0.0.0", port=10000)
