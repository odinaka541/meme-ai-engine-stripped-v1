
"""
clean/extract/summarize the text

cleaning the extracted text here, mainly
"""

# imports -----
import re, wordninja

# a simple system to clean text using regex, then separate the word segs
def clean_extracted_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # text = re.sub(r"[^\x00-\x7F]+", "", text)
    # text = re.sub(r"[^\w\s,.-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace

    text = text.replace(" ", "")

    if not text:
        return ""
    try:
        segments = wordninja.split(text.lower())
        if not all(isinstance(word, str) for word in segments):
            return ""
        return " ".join(segments)
    except Exception as e:
        print("Wordninja is failing", e)
        return ""