# Campus FAQ Chatbot — Assignment Version

This project is a multilingual campus FAQ chatbot built for an AI assignment.
It uses a shared FAQ dataset and a unified Streamlit front end so the project
looks like a complete prototype while still being easy to extend.

| Method | Approach | Owner | Status |
|---|---|---|---|
| A | TF-IDF + SVM (traditional ML) | You | ✅ implemented |
| B | Multilingual BERT (deep learning) | Teammate B | 🔲 placeholder |
| C | Rasa (platform-based) | Teammate C | 🔲 placeholder |

## What is already working

- Shared question dataset in `data/faq_data.csv`
- Canned responses in `data/responses.py`
- TF-IDF + SVM intent classifier in `train_method_a.py`
- Streamlit app UI with single-method and compare-all modes
- Quick question buttons and chat history for a smoother demo
- Graceful fallback when model files are missing

## Project structure

```
Campus-FAQ-Chatbot/
├── app.py
├── train_method_a.py
├── requirements.txt
├── README.md
├── data/
│   ├── faq_data.csv
│   └── responses.py
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
streamlit run app.py
```

## Demo features

- Ask a question in single-method mode
- Compare all three methods side by side
- Quick example buttons for common FAQ questions
- Conversation history in the chat area
- Fallback reply when confidence is low or intent is unknown

## How to extend it

1. Add more examples to `data/faq_data.csv` for better coverage.
2. Retrain Method A using `python train_method_a.py`.
3. Replace the placeholder functions in `app.py` for Method B and Method C.
4. Keep the same intent names across the team so comparison is fair.

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
