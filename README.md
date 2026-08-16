# Campus FAQ Chatbot — Unified Prototype

Multilingual (English / Bahasa Malaysia / Mandarin / Manglish) intent-based
FAQ chatbot for a university context. Three teammates, three different
intent-recognition methods, one shared Streamlit demo.

| Method | Approach | Owner | Status |
|---|---|---|---|
| A | TF-IDF + SVM (traditional ML) | You | ✅ implemented |
| B | Multilingual BERT (deep learning) | Teammate B | 🔲 TODO (stub in `app.py`) |
| C | Rasa (platform-based) | Teammate C | 🔲 TODO (stub in `app.py`) |

## Project structure

```
campus_chatbot/
├── data/
│   ├── faq_data.csv        # shared training data (all 3 methods should use the same intents)
│   └── responses.py        # intent -> canned reply lookup, shared by all methods
├── models/                 # trained model artifacts get saved here
├── train_method_a.py       # trains Method A (TF-IDF + SVM) and saves it to models/
├── app.py                  # unified Streamlit interface (single method / compare all 3)
└── requirements.txt
```

## How to run (Method A only, works right now)

```bash
pip install -r requirements.txt
python train_method_a.py      # trains and saves the model + evaluation report
streamlit run app.py          # opens the chatbot in your browser
```

## How teammates plug in their method

Both `predict_method_b()` and `predict_method_c()` in `app.py` are stubs.
Each must return a tuple: `(intent: str, confidence: float)` — same format
as `predict_method_a()`. As long as that contract is kept, the UI (single
method view + side-by-side comparison view) will work automatically without
any changes to `app.py`'s UI code.

**Important: everyone should train/build on the same `data/faq_data.csv`
intents** (`exam_timetable`, `course_registration`, `fee_payment`,
`hostel_application`, `library_service`, `it_support`) so the final
comparison in the documentation (accuracy, F1, confidence, response style)
is a fair like-for-like comparison across methods.

### Teammate B — BERT
1. Fine-tune `bert-base-multilingual-cased` or `xlm-roberta-base` on
   `data/faq_data.csv`.
2. Save the model to `models/method_b_bert/`.
3. Fill in `predict_method_b()` in `app.py` (a commented example is already
   there).

### Teammate C — Rasa
1. Build a separate Rasa project (`nlu.yml`, `domain.yml`, `stories.yml`)
   using the same intents.
2. Run it locally: `rasa run --enable-api --cors "*" --port 5005`.
3. Fill in `predict_method_c()` in `app.py` to call Rasa's `/model/parse`
   REST endpoint (a commented example is already there).

## Extending the dataset

`data/faq_data.csv` currently has ~70 short examples across 6 intents —
enough to get a working baseline, but you'll want more data (and more
Manglish/code-switching variety) for a stronger "Excellent" grade. Add rows
directly to the CSV; no code changes needed, just re-run `train_method_a.py`.

## Evaluation

Running `train_method_a.py` writes `models/method_a_eval_report.txt`
containing accuracy, precision, recall, F1 (per intent) and a confusion
matrix — paste this into the documentation's Methodology/Results section.
Do the same for Methods B and C so the final comparison table is easy to
build.
