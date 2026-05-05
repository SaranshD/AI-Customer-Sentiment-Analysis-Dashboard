import re

_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
_MENTION_HASHTAG_PATTERN = re.compile(r'[@#]\w+')
_WHITESPACE_PATTERN = re.compile(r'\s+')
_NON_WORD_PATTERN = re.compile(r'[^\w\s]')


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_PATTERN.sub('', text)
    text = _MENTION_HASHTAG_PATTERN.sub('', text)
    text = _NON_WORD_PATTERN.sub('', text)
    return _WHITESPACE_PATTERN.sub(' ', text).strip()
