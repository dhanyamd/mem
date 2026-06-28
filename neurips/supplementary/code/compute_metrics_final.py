import json
import os
import re
import string
from collections import Counter, defaultdict

def normalize_text(text):
    """Lowercases, removes punctuation and articles."""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return " ".join(text.split())

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_em(prediction, ground_truth):
    return float(normalize_text(prediction) == normalize_text(ground_truth))

def main():
    target_file = ".//locomo_CONV26_FIXED.jsonl"
    if not os.path.exists(target_file):
        print(f"❌ File not found: {target_file}")
        return

    CAT_NAMES = {
        1: "Single-hop",
        2: "Multi-hop",
        3: "Open-domain",
        4: "Temporal",
        5: "Adversarial"
    }

    cat_f1 = defaultdict(list)
    cat_em = defaultdict(list)
    conv_f1 = defaultdict(list)
    all_f1 = []
    all_em = []

    with open(target_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line)
                f1 = compute_f1(data["hypothesis"], data["answer"])
                em = compute_em(data["hypothesis"], data["answer"])
                cat = data.get("category", 0)
                conv = data.get("conv_id", "unknown")

                cat_f1[cat].append(f1)
                cat_em[cat].append(em)
                conv_f1[conv].append(f1)
                all_f1.append(f1)
                all_em.append(em)
            except:
                continue

    if not all_f1:
        print("No results found.")
        return

    print("\n" + "=" * 55)
    print("  🏆 RADIX-TITAN LoCoMo BENCHMARK RESULTS 🏆")
    print("=" * 55)

    print("\n📊 BY CATEGORY:")
    print("-" * 55)
    for cat_id in sorted(cat_f1.keys()):
        name = CAT_NAMES.get(cat_id, f"Category {cat_id}")
        avg_f1 = sum(cat_f1[cat_id]) / len(cat_f1[cat_id])
        avg_em = sum(cat_em[cat_id]) / len(cat_em[cat_id])
        n = len(cat_f1[cat_id])
        print(f"  {name:<20} F1: {avg_f1*100:>6.2f}%  EM: {avg_em*100:>5.1f}%  (n={n})")

    print(f"\n📊 BY CONVERSATION:")
    print("-" * 55)
    for conv in sorted(conv_f1.keys()):
        avg = sum(conv_f1[conv]) / len(conv_f1[conv])
        n = len(conv_f1[conv])
        print(f"  {conv:<12} F1: {avg*100:>6.2f}%  (n={n})")

    global_f1 = sum(all_f1) / len(all_f1)
    global_em = sum(all_em) / len(all_em)

    print("\n" + "=" * 55)
    print(f"  🚀 OVERALL F1:          {global_f1*100:.2f}%")
    print(f"  🎯 OVERALL EXACT MATCH: {global_em*100:.2f}%")
    print(f"  📝 TOTAL QUESTIONS:     {len(all_f1)}")
    print("=" * 55 + "\n")

    report = {
        "overall_f1": round(global_f1, 4),
        "overall_em": round(global_em, 4),
        "total_questions": len(all_f1),
        "by_category": {
            CAT_NAMES.get(k, f"cat_{k}"): {
                "f1": round(sum(v)/len(v), 4),
                "em": round(sum(cat_em[k])/len(cat_em[k]), 4),
                "count": len(v)
            } for k, v in sorted(cat_f1.items())
        },
        "by_conversation": {
            k: {"f1": round(sum(v)/len(v), 4), "count": len(v)}
            for k, v in sorted(conv_f1.items())
        }
    }
    with open("final_metrics_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("✅ Full report saved to final_metrics_report.json")

if __name__ == "__main__":
    main()
