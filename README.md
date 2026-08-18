# Campus FAQ Chatbot — Assignment Version

This project is a multilingual campus FAQ chatbot built for an AI assignment.
It uses a shared FAQ dataset and a unified Streamlit front end so the project
looks like a complete prototype while still being easy to extend.

| Method | Approach | Owner | Status |
|---|---|---|---|
| A | TF-IDF + SVM (traditional ML) | You | ✅ implemented |
| B | Multilingual BERT (deep learning) | Teammate B | ✅ implemented (train script included) |
| C | Rasa (platform-based) | Teammate C | ✅ implemented (Rasa project scaffold included) |

## What is already working

- Shared question dataset in `data/faq_data.csv`
- Canned responses in `data/responses.py`
- TF-IDF + SVM intent classifier in `train_method_a.py`
- Multilingual BERT training pipeline in `train_method_b.py`
- Rasa training scaffold in `rasa_project/`
- Streamlit app UI with single-method and compare-all modes
- Quick question buttons and chat history for a smoother demo
- Graceful fallback when model files are missing

## Project structure

```
Campus-FAQ-Chatbot/
├── app.py
├── train_method_a.py
├── train_method_b.py
├── requirements.txt
├── README.md
├── data/
│   ├── faq_data.csv
│   └── responses.py
├── rasa_project/
│   ├── config.yml
│   ├── domain.yml
│   ├── credentials.yml
│   ├── endpoints.yml
│   ├── actions/
│   │   └── __init__.py
│   └── data/
│       ├── nlu.yml
│       └── rules.yml
├── models/
│   ├── method_a_vectorizer.pkl
│   ├── method_a_svm.pkl
│   ├── method_a_label_encoder.pkl
│   └── method_a_eval_report.txt
└── .gitignore
```

## Run the project

```bash
pip install -r requirements.txt
python train_method_a.py
python train_method_b.py
streamlit run app.py
```

## Method B (multilingual BERT)

Train and save the Method B model:

```bash
python train_method_b.py --epochs 3 --batch-size 8
```

After training, the model artifacts are saved to `models/method_b_bert/` and are loaded automatically by `app.py`.

The saved BERT weights are larger than GitHub's normal 100 MB file limit. This repository includes Git LFS tracking for them. Before pushing, run:

```bash
git lfs install
git add .gitattributes models/method_b_bert
git add .
git commit -m "Add trained Method B model"
git push
```

## Method C (Rasa)

Install Rasa separately (optional dependency), then train and run server:

```bash
pip install rasa
cd rasa_project
rasa train
rasa run --enable-api --cors "*" --port 5005
```

`app.py` calls `http://localhost:5005/model/parse` by default, so keep that Rasa server running during a local Method C demo. For Streamlit Cloud, deploy Rasa as a separate public service and set its parse endpoint through `RASA_URL`, for example `https://your-rasa-service.example.com/model/parse`.

## Demo features

- Ask a question in single-method mode
- Compare all three methods side by side
- Quick example buttons for common FAQ questions
- Conversation history in the chat area
- Fallback reply when confidence is low or intent is unknown

## How to extend it

1. Add more examples to `data/faq_data.csv` for better coverage.
2. Retrain Method A using `python train_method_a.py`.
3. Improve `train_method_b.py` hyperparameters or dataset size for better Method B accuracy.
4. Expand `rasa_project/data/nlu.yml` and retrain Rasa for stronger Method C intent quality.
5. Keep the same intent names across the team so comparison is fair.

## Common intents

- `exam_timetable`
- `course_registration`
- `fee_payment`
- `hostel_application`
- `library_service`
- `it_support`

## Assignment tips

- Show the final app in demo mode and explain the pipeline clearly.
- Mention that the shared dataset is important for fair comparison.
- Include a short note on confidence threshold and fallback behavior.
- Add model evaluation metrics from `models/method_a_eval_report.txt` in your report.
