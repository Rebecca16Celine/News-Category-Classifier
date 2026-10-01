import json
import pandas as pd
from collections import Counter

INPUT_FILE = "data/News_Category_Dataset_v3.json"
OUTPUT_FILE = "data/news_balanced.csv"

SAMPLES_PER_CATEGORY = 4261

# Categories we want
CATEGORY_MAP = {
    "POLITICS": "politics",
    "ENTERTAINMENT": "entertainment",
    "BUSINESS": "business",
    "SPORTS": "sports",
    "TRAVEL": "travel",
    "WELLNESS": "health_wellness",
    "HEALTHY LIVING": "health_wellness",
    "STYLE & BEAUTY": "style_beauty",
    "PARENTING": "parenting",
    "FOOD & DRINK": "food_drink",
}    

# SCIENCE + TECH will be combined
COMBINED_SCI_TECH = {
    "SCIENCE",
    "TECH"
}

rows = []

print("Loading dataset...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        category = item["category"]

        # Combine Science and Tech
        if category in COMBINED_SCI_TECH:
            new_category = "science_technology"

        elif category in CATEGORY_MAP:
            new_category = CATEGORY_MAP[category]

        else:
            continue

        # Combine headline + description
        headline = item.get("headline", "").strip()
        description = item.get("short_description", "").strip()

        text = f"{headline} {description}".strip()

        if text:
            rows.append({
                "text": text,
                "category": new_category
            })

print(f"Relevant articles loaded: {len(rows):,}")

df = pd.DataFrame(rows)

# Remove exact duplicate texts
df = df.drop_duplicates(subset=["text"])

print(f"After removing duplicates: {len(df):,}")

# Take exactly 4,310 from each category
balanced_parts = []

for category in sorted(df["category"].unique()):
    category_df = df[df["category"] == category]

    print(f"{category}: {len(category_df):,} available")

    if len(category_df) < SAMPLES_PER_CATEGORY:
        print(f"ERROR: Not enough samples for {category}")
        raise ValueError(f"Not enough samples for {category}")

    sampled = category_df.sample(
        n=SAMPLES_PER_CATEGORY,
        random_state=42
    )

    balanced_parts.append(sampled)

# Combine all categories
balanced_df = pd.concat(balanced_parts, ignore_index=True)

# Shuffle
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save
balanced_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nDataset created successfully!")
print(f"Total articles: {len(balanced_df):,}")
print(f"Number of categories: {balanced_df['category'].nunique()}")

print("\nCategory distribution:")
print(balanced_df["category"].value_counts().sort_index())

print(f"\nSaved to: {OUTPUT_FILE}")