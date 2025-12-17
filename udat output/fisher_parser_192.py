import os
import re
from bs4 import BeautifulSoup
import pandas as pd

HTML_FOLDER = r"C:\Users\Prath\OneDrive\Desktop\Documents\GitHub\Data_Science_Foundation_Final_Project\udat output\html"
FILE = "result_0.8.html"

def extract_split_tables(html_path):
    with open(html_path, "r", encoding="latin-1", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    tables = {}
    for split_id in range(11):  # split0 to split10
        table = soup.find("table", {"id": f"FeaturesUsed_split{split_id}"})
        if table:
            rows = table.find_all("tr")
            tables[split_id] = [r.text.strip() for r in rows[:192]]  # all 192 features, parsing

    return tables

def parse_feature_row(row):
    match = re.match(r"\d+\.\s*(.*?):\s*([0-9.]+)", row)
    if not match:
        return None, None
    return match.group(1).strip(), float(match.group(2))

def aggregate_fisher():
    html_path = os.path.join(HTML_FOLDER, FILE)
    tables = extract_split_tables(html_path)

    score_map = {}

    for split, rows in tables.items():
        for row in rows:
            feat, score = parse_feature_row(row)
            if feat:
                score_map.setdefault(feat, []).append(score)

    # average all collected scores
    data = []
    for feat, scores in score_map.items():
        avg = sum(scores) / len(scores)
        data.append((feat, avg))

    df = pd.DataFrame(data, columns=["feature", "avg_fisher"])
    df = df.sort_values("avg_fisher", ascending=False)
    return df

df = aggregate_fisher()
print(df)
df.to_csv("top10_fisher_avg.csv", index=False)
