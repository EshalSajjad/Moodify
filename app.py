import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── load & prep data once at startup ─────────────────────────────────────────
df = pd.read_csv("cleaned_dataset_spotify.csv")

FEATURE_COLS = ["energy", "danceability", "valence", "loudness", "acousticness"]
CLUSTER_NAMES = {
    0: "Energetic Instrumental EDM",
    1: "Happy Pop / Dance",
    2: "Mainstream High-Energy Pop",
    3: "Sad Acoustic (Instrumental)",
    4: "Sad Acoustic (Vocal)",
}
# CLUSTER_EMOJIS = {
#     0: "⚡",
#     1: "💃",
#     2: "🎉",
#     3: "🎸",
#     4: "🥺",
# }

# ── train Random Forest mood classifier ──────────────────────────────────────
# same features as your notebook Cell 26
RF_FEATURES = ["danceability", "energy", "loudness", "speechiness",
               "acousticness", "instrumentalness", "liveness", "valence", "tempo"]

_rf_data = df[RF_FEATURES + ["cluster"]].dropna()
X_rf = _rf_data[RF_FEATURES]
y_rf = _rf_data["cluster"]
X_train, X_test, y_train, y_test = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42, stratify=y_rf
)
mood_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
mood_classifier.fit(X_train, y_train)
print(f"Mood classifier ready. Accuracy: {mood_classifier.score(X_test, y_test):.2%}")


def predict_mood(row):
    """Use the RF classifier to predict mood from audio features."""
    features = pd.DataFrame([{
        "danceability": row.get("danceability", 0),
        "energy": row.get("energy", 0),
        "loudness": row.get("loudness", 0),
        "speechiness": row.get("speechiness", 0),
        "acousticness": row.get("acousticness", 0),
        "instrumentalness": row.get("instrumentalness", 0),
        "liveness": row.get("liveness", 0),
        "valence": row.get("valence", 0),
        "tempo": row.get("tempo", 0),
    }])
    return int(mood_classifier.predict(features)[0])


# ── similarity setup (sample for performance) ────────────────────────────────
df_sample = df.sample(n=8000, random_state=42).reset_index(drop=True)

X = df_sample[FEATURE_COLS].fillna(df_sample[FEATURE_COLS].mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
similarity_matrix = cosine_similarity(X_scaled)


def recommend(song_idx, top_n=8, popularity_weight=0.2):
    sim_scores = similarity_matrix[song_idx]
    pop_norm = df_sample["popularity"] / 100.0
    final_scores = (1 - popularity_weight) * sim_scores + popularity_weight * pop_norm.values
    top_indices = np.argsort(final_scores)[::-1][1:top_n + 1]
    recs = df_sample.iloc[top_indices].copy()
    recs["similarity"] = (sim_scores[top_indices] * 100).round(1)
    return recs


def song_to_dict(row, similarity=None):
    cluster_id = predict_mood(row)
    d = {
        "track_name": row["track_name"],
        "track_genre": row["track_genre"],
        "popularity": int(row["popularity"]),
        "cluster": cluster_id,
        "cluster_name": CLUSTER_NAMES.get(cluster_id, "Unknown"),
        # "cluster_emoji": CLUSTER_EMOJIS.get(cluster_id, "🎵"),
        "energy": round(float(row["energy"]), 2),
        "valence": round(float(row["valence"]), 2),
        "danceability": round(float(row["danceability"]), 2),
    }
    if similarity is not None:
        d["similarity"] = float(similarity)
    return d


# ── routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get("q", "").strip().lower()
    if len(query) < 2:
        return jsonify([])
    matches = df_sample[df_sample["track_name"].str.lower().str.contains(query, na=False)]
    results = (
        matches[["track_name", "track_genre", "popularity"]]
        .drop_duplicates("track_name")
        .sort_values("popularity", ascending=False)
        .head(8)
    )
    return jsonify(results.reset_index().rename(columns={"index": "idx"}).to_dict("records"))


@app.route("/recommend/<int:idx>")
def get_recommendations(idx):
    if idx >= len(df_sample):
        return jsonify({"error": "Song not found"}), 404

    input_song = df_sample.iloc[idx]
    recs = recommend(idx)

    return jsonify({
        "input": song_to_dict(input_song),
        "recommendations": [
            song_to_dict(row, similarity=row["similarity"])
            for _, row in recs.iterrows()
        ],
    })


if __name__ == "__main__":
    app.run(debug=True)
