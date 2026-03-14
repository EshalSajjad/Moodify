# Moodify
## Spotify Song Recommender

A mood-based song recommender built with ML. Search any song, get 8 recommendations based on audio feature similarity and mood clustering.

## How it works

1. **K-Means clustering** groups 100k+ songs into 5 mood categories based on audio features (energy, valence, danceability, acousticness, instrumentalness)
2. **Cosine similarity** finds songs with the closest audio fingerprint to your input
3. **Popularity weighting** blends similarity with popularity score for better results

### Mood clusters
| Cluster Name |
|---|
| Energetic Instrumental EDM |
| Happy Pop / Dance |
| Mainstream High-Energy Pop |
| Sad Acoustic (Instrumental) |
| Sad Acoustic (Vocal) |

## Tech stack
- **Backend:** Flask, scikit-learn, pandas, numpy
- **ML:** K-Means, PCA, cosine similarity, Random Forest classifier
- **Deployment:** Render

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Dataset
Spotify audio features dataset — 106k tracks across 114 genres.
Features used: `energy`, `danceability`, `valence`, `loudness`, `acousticness`, `speechiness`, `instrumentalness`, `tempo`
