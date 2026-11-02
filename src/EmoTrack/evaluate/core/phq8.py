from __future__ import annotations


PHQ8_ITEM_TEXT_BY_SYMPTOM = {
    "anhedonia": "Little interest or pleasure in doing things",
    "depressed_mood": "Feeling down, depressed, or hopeless",
    "sleep": "Trouble falling or staying asleep, or sleeping too much",
    "fatigue": "Feeling tired or having little energy",
    "appetite": "Poor appetite or overeating",
    "self_worth": (
        "Feeling bad about yourself, or that you are a failure, "
        "or have let yourself or your family down"
    ),
    "concentration": (
        "Trouble concentrating on things, such as reading the newspaper "
        "or watching television"
    ),
    "psychomotor": (
        "Moving or speaking so slowly that other people could have noticed. "
        "Or the opposite - being so fidgety or restless that you have been "
        "moving around a lot more than usual"
    ),
}

DEFAULT_PHQ8_SYMPTOM_ORDER = tuple(PHQ8_ITEM_TEXT_BY_SYMPTOM)
