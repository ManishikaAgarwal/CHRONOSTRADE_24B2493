import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt

# =============================
# LOAD DATA
# =============================

df = pd.read_csv("dataset.csv")   # rename based on file

print(df.head())
print(df.columns)

TEXT_COL = "review"   # change if different

# =============================
# SENTIMENT FUNCTION
# =============================

def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"

# =============================
# APPLY
# =============================

df["Sentiment"] = df[TEXT_COL].apply(get_sentiment)

# =============================
# DISTRIBUTION
# =============================

counts = df["Sentiment"].value_counts()
percent = df["Sentiment"].value_counts(normalize=True) * 100

summary = pd.DataFrame({
    "Count": counts,
    "Percentage": percent.round(2)
})

print(summary)

summary.to_csv("sentiment_distribution.csv")

# =============================
# SAVE DATASET
# =============================

df.to_csv("dataset_with_sentiment.csv", index=False)

# =============================
# VISUALIZE
# =============================

plt.figure(figsize=(6,5))
counts.plot(kind="bar")
plt.title("Sentiment Distribution")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
