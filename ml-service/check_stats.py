import json
from collections import Counter

with open("data/levir_mci/LEVIR-MCI-dataset/LevirCCcaptions.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

train_items = [x for x in meta["images"] if x["split"] == "train"]
change_flags = Counter(x.get("changeflag", 1) for x in train_items)
print("Train change flags (0=no change, 1=change):", change_flags)

all_sentences = []
for x in train_items:
    for s in x["sentences"]:
        all_sentences.append(s["raw"].strip().lower())

print("Total train sentences:", len(all_sentences))
c = Counter(all_sentences)
print("Top 8 most frequent sentences:")
for s, count in c.most_common(8):
    print(f"  {count} ({count/len(all_sentences)*100:.1f}%): \"{s}\"")
