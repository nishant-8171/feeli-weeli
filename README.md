# AI Emotion Detection and Support System

A Python Streamlit app that detects user emotions using a pre-trained RoBERTa model and uses Gemini API to generate emotional understanding, motivational support, a correct solution, and a video recommendation topic.

## Project Flow

```text
User sentence
↓
Pre-trained RoBERTa emotion detection
↓
Top 3 emotions with confidence score
↓
Gemini API emotional understanding
↓
Motivational support and correct solution
↓
YouTube video recommendation topic
```

## Files

```text
app.py
requirements.txt
runtime.txt
README.md
.gitignore
.streamlit/config.toml
.streamlit/secrets.toml.example
```

## Run Locally

Open CMD and run:

```cmd
cd C:\Users\nisha\Documents\Codex\2026-05-16\this-it-the-work-flow-of
py -m pip install -r requirements.txt
set GEMINI_API_KEY=your_real_new_gemini_api_key
py -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

You can also paste your Gemini API key into the sidebar password field for testing.

## Deploy On Streamlit Cloud

1. Upload these files to a GitHub repository.
2. Go to Streamlit Cloud.
3. Click New app.
4. Select your GitHub repository.
5. Set main file path:

```text
app.py
```

6. Open Advanced settings or App secrets.
7. Add this secret:

```toml
GEMINI_API_KEY = "your_real_new_gemini_api_key"
```

8. Click Deploy.

## Important

Do not write your API key directly inside `app.py`.
Do not upload `.streamlit/secrets.toml` to GitHub.
If an API key was shared publicly, delete it and create a new key.

## Safety Note

This app gives emotional support only. It does not provide medical diagnosis. If someone is in immediate danger, they should contact emergency help or a trusted person.
