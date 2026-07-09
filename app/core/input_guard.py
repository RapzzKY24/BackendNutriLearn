import re

INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|directions|prompts)",
    r"(?i)(reveal|show|print|display|output|leak|dump)\s+(your|the|this)\s+(system\s+)?prompt",
    r"(?i)forget\s+(all\s+)?(previous|prior)\s+(instructions|training|context)",
    r"(?i)you\s+(are\s+)?(now|will\s+act\s+as)\s+(DAN|jailbreak|unfiltered|unguided)",
    r"(?i)role\s*[Pp]lay\s+as",
    r"(?i)new\s+rule",
]


def sanitize_input(text: str) -> str:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            raise ValueError("Input contains prohibited content")
    return text.strip()


def guard_question(question: str) -> str:
    return sanitize_input(question)
