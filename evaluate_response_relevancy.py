"""
Evaluate response relevancy for Method A/B/C using BLEU and ROUGE-L.

Important caveat:
    This chatbot answers with fixed canned responses per intent (see data/responses.py),
    not free-text generation. So BLEU/ROUGE here mainly measure whether each method
    retrieved the CORRECT canned response for a question - i.e. they reflect intent
    classification correctness translated into response text, not generative language
    quality. A wrong-intent prediction produces a completely different reference
    response, which is why the score drops sharply on misclassified questions.

Usage:
    python evaluate_response_relevancy.py
    (run evaluate_methods.py first to produce data/evaluation_results.csv)

Writes:
    models/response_relevancy_report.txt
"""

from pathlib import Path

import nltk
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from data.responses import RESPONSES

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "data" / "evaluation_results.csv"
REPORT_PATH = BASE_DIR / "models" / "response_relevancy_report.txt"

METHOD_COLUMNS = {
    "Method A (TF-IDF + SVM)": "method_a_pred",
    "Method B (multilingual BERT)": "method_b_pred",
    "Method C (Rasa)": "method_c_pred",
}


def get_response(intent: str) -> str:
    return RESPONSES.get(intent, RESPONSES["fallback"])


def score_pair(reference: str, candidate: str, scorer: rouge_scorer.RougeScorer, smoothing) -> tuple[float, float]:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    bleu = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
    rouge_l = scorer.score(reference, candidate)["rougeL"].fmeasure
    return bleu, rouge_l


def main():
    nltk.download("punkt", quiet=True)

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} not found. Run evaluate_methods.py first.")

    df = pd.read_csv(RESULTS_PATH)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    report_lines = [
        "Response Relevancy Evaluation (BLEU / ROUGE-L)",
        "",
        "Caveat: responses are fixed canned text per intent, so these scores mainly reflect",
        "whether each method retrieved the correct intent's response, not generation quality.",
        "",
        f"{'Method':<30}{'Avg BLEU':>12}{'Avg ROUGE-L':>14}",
    ]

    for method_name, pred_col in METHOD_COLUMNS.items():
        bleu_scores = []
        rouge_scores = []
        for _, row in df.iterrows():
            reference = get_response(row["expected_intent"])
            candidate = get_response(row[pred_col])
            bleu, rouge_l = score_pair(reference, candidate, scorer, smoothing)
            bleu_scores.append(bleu)
            rouge_scores.append(rouge_l)

        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        report_lines.append(f"{method_name:<30}{avg_bleu:>12.4f}{avg_rouge:>14.4f}")
        print(f"{method_name}: avg_bleu={avg_bleu:.4f} avg_rouge_l={avg_rouge:.4f}")

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
