"""
intent.py
---------
Intent classification using GPT (primary) with a rule-based fallback.

Primary path:  GPT-4o-mini classifies intent from the transcript + emotion.
Fallback path: keyword scoring + optional zero-shot HuggingFace model
               (same as before) — used if the OpenAI call fails.
"""

import os
import re
import json
from typing import Optional

# ── Intent labels ─────────────────────────────────────────────────────────────
INTENT_LABELS = ["aggressive", "informative", "promotional", "conversational"]

INTENT_EXPLANATIONS = {
    "aggressive": (
        "The speaker appears to have an aggressive intent. The language used "
        "in this video contains confrontational, hostile, or forceful expressions "
        "that suggest the speaker is trying to challenge, attack, or dominate the "
        "listener. This could include strong criticism, offensive remarks, threats, "
        "or emotionally charged arguments directed at a person, group, or idea. "
        "Viewers should approach this content with caution."
    ),
    "informative": (
        "The speaker demonstrates an informative intent. The content of this video "
        "is focused on educating, explaining, or reporting facts to the audience. "
        "The speaker uses structured language to present knowledge, describe a "
        "process, share research findings, or provide updates on a topic. The goal "
        "appears to be increasing the viewer's understanding rather than persuading "
        "or entertaining them."
    ),
    "promotional": (
        "The speaker exhibits a promotional intent. The language and tone in this "
        "video are designed to persuade the audience to take a specific action — "
        "such as buying a product, subscribing to a service, or supporting a cause. "
        "The content may include calls-to-action, benefit-focused language, limited-time "
        "offers, or endorsements. The primary goal is to influence the viewer's "
        "decision or behaviour in favour of something being advertised."
    ),
    "conversational": (
        "The speaker has a conversational intent. The tone of this video is casual, "
        "personal, and interactive, as if the speaker is having a friendly discussion "
        "with the audience. The content may include storytelling, sharing personal "
        "opinions, asking questions, or simply chatting about everyday topics. There "
        "is no strong agenda to sell, teach, or confront — the primary goal is "
        "natural, relatable communication and connection with the viewer."
    ),
}

# ── Keyword rules (used as fallback only) ─────────────────────────────────────
INTENT_KEYWORDS: dict[str, list[str]] = {
    "aggressive": [
        "attack", "fight", "hate", "kill", "stupid", "idiot", "shut up",
        "useless", "damn", "hell", "angry", "rage", "war", "enemy",
        "destroy", "threaten", "abuse", "curse", "violent", "hodi", "sala",
    ],
    "promotional": [
        "buy", "sale", "offer", "discount", "subscribe", "click", "link",
        "deal", "limited time", "free", "win", "earn", "coupon", "promo",
        "shop", "price", "order", "checkout", "sponsored", "ad", "brand",
        "product", "service", "kharidi", "belade",
    ],
    "informative": [
        "explain", "learn", "tutorial", "how to", "guide", "introduction",
        "definition", "history", "fact", "research", "study", "analysis",
        "news", "report", "update", "knowledge", "educate", "information",
        "review", "comparison", "science", "technology", "discover",
        "tilisi", "kalisi", "maahiti", "vyaakhyana",
    ],
    "conversational": [
        "hi", "hello", "hey", "how are you", "what do you think",
        "let me know", "talk", "chat", "share", "my opinion", "i think",
        "today i", "story", "fun", "life", "friend", "family", "love",
        "feel", "experience", "personal", "hEge idira", "naanu",
    ],
}


# ── GPT-based intent detection ─────────────────────────────────────────────────
def _detect_intent_gpt(transcript: str,
                       emotion: Optional[str] = None,
                       api_key: Optional[str] = None
                       ) -> tuple[str, dict[str, float], str]:
    """
    Classify intent using GPT-4o-mini.
    Returns (intent_label, probabilities_dict, explanation_text).
    Raises on failure so caller can fall back.
    """
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai package not installed")

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = openai.OpenAI(api_key=key)

    emotion_line = f"\nSpeaker's detected emotion: {emotion}" if emotion else ""

    prompt = f"""You are an intent classifier for video content analysis.

Analyse the following video transcript and classify the speaker's intent.{emotion_line}

Transcript:
\"\"\"
{transcript[:3000]}
\"\"\"

Classify the intent into exactly ONE of these categories:
- aggressive: confrontational, hostile, or forceful expressions; attacks, insults, threats
- informative: educational, explanatory, or fact-based content meant to teach or report
- promotional: persuasive content designed to sell, advertise, or promote something
- conversational: casual, personal, friendly dialogue; storytelling or sharing opinions

Respond with a JSON object with exactly these keys:
{{
  "intent": "<one of: aggressive | informative | promotional | conversational>",
  "probabilities": {{
    "aggressive": <float 0-1>,
    "informative": <float 0-1>,
    "promotional": <float 0-1>,
    "conversational": <float 0-1>
  }},
  "explanation": "<2–3 sentences explaining why this intent was detected, citing specific cues from the transcript>"
}}
The probabilities must sum to 1.0."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=400,
        temperature=0.2,
    )

    raw = json.loads(response.choices[0].message.content)

    intent = raw.get("intent", "conversational")
    if intent not in INTENT_LABELS:
        intent = "conversational"

    probs = {k: float(raw.get("probabilities", {}).get(k, 0.25))
             for k in INTENT_LABELS}
    total = sum(probs.values()) or 1.0
    probs = {k: round(v / total, 4) for k, v in probs.items()}

    explanation = raw.get("explanation", INTENT_EXPLANATIONS[intent])

    return intent, probs, explanation


# ── Rule-based fallback ────────────────────────────────────────────────────────
def _rule_based_scores(text: str) -> dict[str, float]:
    text_lower = text.lower()
    scores: dict[str, float] = {k: 0.0 for k in INTENT_KEYWORDS}
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[intent] += 1.0
                if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                    scores[intent] += 0.5
    return scores


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    import math
    vals  = list(scores.values())
    max_v = max(vals)
    exps  = [math.exp(v - max_v) for v in vals]
    total = sum(exps)
    return {k: exps[i] / total for i, k in enumerate(scores.keys())}


_zs_classifier = None


def _get_zero_shot():
    global _zs_classifier
    if _zs_classifier is None:
        try:
            from transformers import pipeline
            _zs_classifier = pipeline(
                "zero-shot-classification",
                model="cross-encoder/nli-MiniLM2-L6-H768",
                device=-1,
            )
        except Exception:
            _zs_classifier = False
    return _zs_classifier if _zs_classifier else None


def _detect_intent_rules(transcript: str,
                         emotion: Optional[str] = None
                         ) -> tuple[str, dict[str, float], str]:
    """Rule-based fallback — same logic as before."""
    if not transcript or not transcript.strip():
        label = "conversational"
        return label, {k: 0.25 for k in INTENT_KEYWORDS}, INTENT_EXPLANATIONS[label]

    scores = _rule_based_scores(transcript)

    if emotion:
        if emotion == "angry":
            scores["aggressive"] += 1.5
        elif emotion == "happy":
            scores["conversational"] += 0.5
            scores["promotional"]    += 0.3
        elif emotion == "neutral":
            scores["informative"] += 0.5

    if max(scores.values()) == 0:
        zs = _get_zero_shot()
        if zs:
            try:
                labels = list(INTENT_KEYWORDS.keys())
                result = zs(transcript[:512], candidate_labels=labels)
                for lbl, sc in zip(result["labels"], result["scores"]):
                    scores[lbl] = sc
            except Exception:
                pass

    if max(scores.values()) == 0:
        scores["conversational"] = 1.0

    probs  = _softmax(scores)
    intent = max(probs, key=probs.get)
    explanation = INTENT_EXPLANATIONS[intent]
    return intent, probs, explanation


# ── Public API ─────────────────────────────────────────────────────────────────
def detect_intent(transcript: str,
                  emotion: Optional[str] = None,
                  api_key: Optional[str] = None
                  ) -> tuple[str, dict[str, float], str]:
    """
    Detect intent from a transcript.

    Tries GPT-4o-mini first; if it fails (no key, network error, etc.)
    falls back silently to the keyword + zero-shot classifier.

    Returns:
        (intent_label, probability_dict, explanation_paragraph)
    """
    # ── Try GPT first ─────────────────────────────────────────────────────────
    try:
        label, probs, explanation = _detect_intent_gpt(transcript, emotion, api_key)
        print(f"[intent] GPT detected intent: {label}")
        return label, probs, explanation
    except Exception as gpt_err:
        print(f"[intent] GPT intent detection failed ({gpt_err}), using rule-based fallback.")

    # ── Rule-based fallback ───────────────────────────────────────────────────
    return _detect_intent_rules(transcript, emotion)
