# Customer Insight Dashboard

https://customer-insightdashboard.streamlit.app/

This Streamlit app helps a company decide which customers to contact first, it combines my models and a text sentiment tool.

## How to run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run this from inside the unzipped folder. You don't need any datasets because the trained models are included.

## What each tab does

- **Customer Churn:** fill in a customer's account details and click "Predict churn risk." It shows the chance they leave, a risk level, and a chart of which details raised or lowered the risk.
- **Text Sentiment:** paste a review or message and click "Analyze text." It shows whether it's positive, neutral or negative, and which words caused that.
- **Audio Sentiment:** upload a short .wav file. It shows whether the call sounds positive or negative, and charts of the audio features behind the prediction.
- **Business Summary:** how the tabs fit together, and the limits of the models.

## Files in this ZIP

| File | What it is |
|---|---|
| `streamlit_app.py` | The dashboard |
| `requirements.txt` | Packages to install |
| `model_utils.py` | Audio model code and feature extraction, used by the app |
| `churn_artifacts.joblib` | Trained churn model from Assignment 1 |
| `audio_model.pt` | Trained audio model from Assignment 4 |
| `preview.png` | Screenshot of the app running |

## Sources

- Churn model: logistic regression.
- Audio model: MFCC features (python_speech_features) into a small PyTorch network.
- Text sentiment: VADER (`vaderSentiment`, Hutto & Gilbert, 2014), a word-list tool, not a model I trained.

## Notes

- The churn model is about 66% accurate, so it's meant to help prioritize who to contact, not make decisions.
- The audio model was tested on only 16 clips, so its results are only a rough guide.
- Audio labels: 0 = Negative, 1 = Positive.
