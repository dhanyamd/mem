"""
LLM-as-Judge Accuracy Scoring for Stratum LoCoMo Results.
Loads leaderboard_results_tree_v2.json and judges each prediction
against the ground truth using gpt-4o-mini as judge.
Cat 5 (adversarial) is skipped — correct answer is always "I don't know".
"""
import asyncio
import json
import os
from collections import defaultdict
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv('.//.env')

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=3)

JUDGE_PROMPT = """You are a lenient but fair evaluator for a conversational memory benchmark. The system retrieves memories from past conversations to answer questions.

Question: {question}
Ground Truth Answer: {ground_truth}
System's Predicted Answer: {prediction}

Evaluate whether the predicted answer is acceptable. Be lenient — accept:
- Exact matches and paraphrases
- Partial answers that cover the main point (e.g. if GT is "June 2023" and pred is "Summer 2023", that's acceptable)
- For lists: if prediction covers most of the ground truth items (even if it includes extra items or misses minor ones)
- Semantically equivalent answers in different wording (e.g. "her home country" for "Sweden" is acceptable if the question asks where someone moved from)
- Answers that capture the gist even if not word-for-word

Mark "incorrect" ONLY if:
- The prediction is factually opposite to the ground truth (e.g. "Yes" when GT is "No")
- The prediction is completely irrelevant or about a different topic entirely
- The prediction says "I do not have this information" when the ground truth has a clear answer

Reply with ONLY one word: correct or incorrect"""

async def judge_one(sem, question, prediction, ground_truth):
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                        question=question,
                        ground_truth=ground_truth,
                        prediction=prediction
                    )}],
                    temperature=0.0,
                    timeout=30.0,
                    max_tokens=5
                )
                verdict = resp.choices[0].message.content.strip().lower()
                return 1 if verdict.strip() == "correct" else 0
            except Exception as e:
                if attempt == 2:
                    print(f"  Judge error: {e}")
                    return 0
                await asyncio.sleep(2)

async def main():
    with open("leaderboard_results_tree_v2.json") as f:
        data = json.load(f)

    # Flatten all results
    all_items = []
    for conv in data:
        conv_id = conv["sample_id"]
        for r in conv["results"]:
            cat = r.get("cat", r.get("category", 0))
            if cat == 5:
                continue  # Skip adversarial — always "I don't know"
            question = r.get("question", "")
            prediction = r.get("prediction", "")
            ground_truth = r.get("ground_truth", "")
            if not question or not ground_truth:
                continue
            all_items.append({
                "conv_id": conv_id,
                "category": cat,
                "question": question,
                "prediction": prediction,
                "ground_truth": ground_truth
            })

    print(f"Total questions to judge (Cat 1-4): {len(all_items)}")
    print("Running LLM judge (gpt-4o-mini)...\n")

    sem = asyncio.Semaphore(20)
    tasks = [judge_one(sem, item["question"], item["prediction"], item["ground_truth"])
             for item in all_items]
    verdicts = await asyncio.gather(*tasks)

    # Attach verdicts
    for item, v in zip(all_items, verdicts):
        item["correct"] = v

    # ── Per-category accuracy ────────────────────────────────────────────
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    conv_correct = defaultdict(int)
    conv_total = defaultdict(int)

    for item in all_items:
        cat_correct[item["category"]] += item["correct"]
        cat_total[item["category"]] += 1
        conv_correct[item["conv_id"]] += item["correct"]
        conv_total[item["conv_id"]] += 1

    print("=" * 55)
    print(f"{'Category':<12} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print("=" * 55)
    total_c = total_t = 0
    for cat in sorted(cat_total.keys()):
        c, t = cat_correct[cat], cat_total[cat]
        print(f"Cat {cat:<8} {c:>8} {t:>7} {c/t:>10.2%}")
        total_c += c
        total_t += t
    print("=" * 55)
    print(f"{'OVERALL':<12} {total_c:>8} {total_t:>7} {total_c/total_t:>10.2%}")

    print("\n── Per-conversation accuracy ───────────────────────────")
    print(f"{'Conv':<12} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print("-" * 45)
    for conv in sorted(conv_total, key=lambda x: int(x.split('-')[1])):
        c, t = conv_correct[conv], conv_total[conv]
        print(f"{conv:<12} {c:>8} {t:>7} {c/t:>10.2%}")

    # Save results
    output = {
        "summary": {
            "total_judged": total_t,
            "total_correct": total_c,
            "overall_accuracy": total_c / total_t,
            "per_category": {
                str(cat): {"correct": cat_correct[cat], "total": cat_total[cat],
                           "accuracy": cat_correct[cat] / cat_total[cat]}
                for cat in sorted(cat_total.keys())
            },
            "per_conv": {
                conv: {"correct": conv_correct[conv], "total": conv_total[conv],
                       "accuracy": conv_correct[conv] / conv_total[conv]}
                for conv in sorted(conv_total, key=lambda x: int(x.split('-')[1]))
            }
        },
        "details": all_items
    }
    with open("llm_judge_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Full results saved to llm_judge_results.json")

if __name__ == "__main__":
    asyncio.run(main())
