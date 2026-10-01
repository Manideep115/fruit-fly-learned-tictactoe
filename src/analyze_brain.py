import pandas as pd

FILE = "data/classification/classification.csv"

df = pd.read_csv(FILE)

print("\n🪰 FLY BRAIN")
print("=" * 50)

print(f"Total classified neurons: {len(df):,}")

print("\n--- Super Classes ---")
print(df["super_class"].value_counts().to_string())

print("\n--- Classes ---")
print(df["class"].value_counts().head(30).to_string())

print("\n--- Flows ---")
print(df["flow"].value_counts().to_string())