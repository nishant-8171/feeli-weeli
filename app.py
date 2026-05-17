import os
from urllib.parse import quote_plus

import streamlit as st
from google import genai
from transformers import pipeline


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
GEMINI_MODEL = "gemini-2.5-flash"


st.set_page_config(
    page_title="AI Emotion Support System",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading pre-trained RoBERTa emotion model...")
def load_emotion_model():
    return pipeline(
        task="text-classification",
        model=MODEL_NAME,
        top_k=None,
    )


def normalize_pipeline_output(raw_result):
    if raw_result and isinstance(raw_result[0], list):
        return raw_result[0]
    return raw_result


def apply_context_correction(user_sentence, results):
    text = user_sentence.lower()

    low_enjoyment_phrases = [
        "not enjoyable",
        "not enjoy",
        "not happy",
        "not fun",
        "not excited",
        "not good",
        "not feeling good",
    ]

    calm_phrases = [
        "not worried",
        "not worry",
        "chilling",
        "calm",
        "relaxed",
    ]

    has_low_enjoyment = any(phrase in text for phrase in low_enjoyment_phrases)
    has_calm_signal = any(phrase in text for phrase in calm_phrases)

    if not has_low_enjoyment:
        return results

    corrected_results = []
    for item in results:
        corrected_item = item.copy()
        label = corrected_item["label"].lower()

        if label == "joy":
            corrected_item["score"] *= 0.35
        elif label == "neutral" and has_calm_signal:
            corrected_item["score"] += 0.45
        elif label == "sadness":
            corrected_item["score"] += 0.08

        corrected_results.append(corrected_item)

    total_score = sum(item["score"] for item in corrected_results)
    if total_score > 0:
        for item in corrected_results:
            item["score"] = item["score"] / total_score

    return corrected_results


def detect_top_3_emotions(user_sentence):
    raw_result = load_emotion_model()(user_sentence)
    results = normalize_pipeline_output(raw_result)
    results = apply_context_correction(user_sentence, results)
    sorted_results = sorted(results, key=lambda item: item["score"], reverse=True)

    return [
        {
            "emotion": item["label"],
            "confidence": round(item["score"] * 100, 2),
        }
        for item in sorted_results[:3]
    ]


def get_api_key(manual_key=None):
    if manual_key:
        return manual_key.strip()

    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()

    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        secret_key = None

    return secret_key.strip() if secret_key else None


def build_prompt(user_sentence, top_3_emotions):
    emotion_lines = "\n".join(
        f"{item['emotion']}: {item['confidence']}%" for item in top_3_emotions
    )

    return f"""
You are an emotional support assistant.

User sentence:
{user_sentence}

Top 3 detected emotions:
{emotion_lines}

Give the answer in this exact format:

Exact Emotional Condition:
AI Emotional Understanding:
Confidence Explanation:
Motivational Support:
Correct Solution:
Video Recommendation Reason:

Rules:
Use simple English.
Do not blindly trust only the emotion percentages.
Understand the full sentence context, especially words like "but", "not", "still", "not enjoyable", and "not worried".
If the user is calm but not enjoying the situation, describe it as neutral, calm, detached, or mildly dissatisfied instead of pure joy.
Do not give medical diagnosis.
Do not say you are a doctor.
Do not make the user feel guilty.
If the user mentions suicide, self-harm, or danger, tell them to contact emergency help or a trusted person immediately.
Keep the answer short, clear, and useful.
"""


def get_gemini_support(user_sentence, top_3_emotions, api_key):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_prompt(user_sentence, top_3_emotions),
    )
    return response.text


def recommend_video(top_3_emotions):
    main_emotion = top_3_emotions[0]["emotion"].lower()

    video_topics = {
        "anger": "calm down anger control motivation",
        "disgust": "emotional healing and positive mindset",
        "fear": "overcome fear and build confidence",
        "joy": "positive life motivation",
        "neutral": "daily motivation and self improvement",
        "sadness": "overcome sadness and find hope",
        "surprise": "mindfulness and emotional balance",
    }

    topic = video_topics.get(main_emotion, "motivational support video")
    youtube_link = "https://www.youtube.com/results?search_query=" + quote_plus(topic)
    return topic, youtube_link


def render_emotion_cards(top_3_emotions):
    columns = st.columns(3)
    for column, item in zip(columns, top_3_emotions):
        column.metric(
            label=item["emotion"].title(),
            value=f"{item['confidence']}%",
        )


st.title("Feeli Weeli")
st.caption("I am Feeli Weeli, your emotion supporter")

with st.sidebar:
    st.header("Emotion Detection Flow")
    st.write("1. Share your feeling")
    st.write("2. See your current feeling")
    st.write("3. Feeli Weeli solution")
    st.write("4. Feeli Weeli video support")


api_key = get_api_key()

user_sentence = st.text_area(
    "Enter user's sentence",
    height=150,
    placeholder="Example: I feel confused because I met my ex after a long time...",
)

analyze = st.button("Analyze Emotion", type="primary", use_container_width=True)

if analyze:
    if not user_sentence.strip():
        st.warning("Please enter a sentence first.")
    elif not api_key:
        st.error("Gemini API key is missing. Set GEMINI_API_KEY in CMD or Streamlit secrets.")
    else:
        try:
            with st.spinner("Detecting emotions with RoBERTa..."):
                top_3_emotions = detect_top_3_emotions(user_sentence)

            st.subheader("Top 3 Emotions")
            render_emotion_cards(top_3_emotions)

            with st.spinner("Generating emotional support with Gemini..."):
                support_result = get_gemini_support(
                    user_sentence=user_sentence,
                    top_3_emotions=top_3_emotions,
                    api_key=api_key,
                )

            video_topic, youtube_link = recommend_video(top_3_emotions)

            st.subheader("AI Emotional Support")
            st.write(support_result)

            st.subheader("Recommended Video")
            st.write(f"Topic: **{video_topic}**")
            st.link_button("Open YouTube Search", youtube_link, use_container_width=True)

            st.info(
                "This app gives emotional support only. It does not provide medical diagnosis. "
                "If someone is in immediate danger, contact local emergency help or a trusted person."
            )
        except Exception as error:
            st.error(f"Something went wrong: {error}")
