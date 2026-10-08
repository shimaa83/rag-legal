
#Data Preprocessing Steps
#  # ============================================================ 
# 1. Load the raw Civil Code JSON dataset.
# 2. Preserve the original Arabic text. 
# 3. Normalize Arabic text for search and embeddings.
# 4. Normalize article numbers to integers. 
# 5. Normalize metadata fields and whitespace. 
# 6. Identify and flag repealed articles.
# 7. Sort articles by article number.
# 8. Save the processed dataset as JSON. 
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from .config import (
    ARABIC_DIGITS,
    ENGLISH_DIGITS,
    PROCESSED_JSON_PATH,
    RAW_JSON_PATH,
    REPEALED_ARTICLES,
)

# ============================================================
# Arabic Text Normalization
# ============================================================


def normalize_arabic_text(text: str) -> str:
    """
    Normalize Arabic text for search and embeddings.

    Operations:
    - Unicode normalization
    - Remove Arabic diacritics
    - Remove Tatweel
    - Normalize Alef variants
    - Normalize Hamza forms
    - Normalize Alef Maqsura
    - Remove punctuation
    - Normalize whitespace

    Note:
        This function creates a normalized copy.
        The original Arabic text is never modified.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)
        # Normalize Lam-Alef ligatures
  # Fix Lam-Alef extraction artifacts
    text = re.sub(
        r"ا([إأآ])ل",
        r"ال\1",
        text,
    )

    text = text.replace(
        "اال",
        "الا",
    )


    # Remove Arabic diacritics / tashkeel
    text = re.sub(
        r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]",
        "",
        text,
    )

    # Remove Tatweel
    text = text.replace("ـ", "")

    # Normalize Alef variants
    text = re.sub(
        r"[إأآٱ]",
        "ا",
        text,
    )

    # Normalize Hamza forms
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")

    # Normalize Alef Maqsura
    text = text.replace("ى", "ي")

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# English Text Normalization
# ============================================================


def normalize_english_text(text: str) -> str:
    """
    Apply light normalization to English legal text.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Unicode normalization
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# Article Number Normalization
# ============================================================


def normalize_article_number(
    value: str | int,
) -> int:
    """
    Convert Arabic-Indic or English article numbers to int.

    Examples:
        "١٤٧" -> 147
        "147" -> 147
        147 -> 147
    """

    if isinstance(value, bool):
        raise TypeError(
            "article number must be a string or integer"
        )

    if isinstance(value, int):
        return value

    if not isinstance(value, str):
        raise TypeError(
            "article number must be a string or integer"
        )

    value = value.strip()

    # Arabic-Indic digits -> English digits
    digit_translation = str.maketrans(
        ARABIC_DIGITS,
        ENGLISH_DIGITS,
    )

    value = value.translate(
        digit_translation
    )

    # Remove whitespace
    value = re.sub(
        r"\s+",
        "",
        value,
    )

    if not value.isdigit():
        raise ValueError(
            f"Invalid article number: {value}"
        )

    return int(value)


# ============================================================
# Repealed Articles
# ============================================================


def is_repealed_article(
    article_number: int,
) -> bool:
    """
    Check whether an article is repealed.
    """

    return article_number in REPEALED_ARTICLES


# ============================================================
# Metadata Normalization
# ============================================================


def normalize_optional_text(
    value: Any,
) -> Any:
    """
    Normalize optional metadata fields.

    None remains None.
    String values are whitespace-normalized.
    Other values are returned unchanged.
    """

    if value is None:
        return None

    if isinstance(value, str):
        value = unicodedata.normalize(
            "NFKC",
            value,
        )

        value = re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

        return value if value else None

    return value


# ============================================================
# Arabic Field Handling
# ============================================================


def get_arabic_text(
    article: dict[str, Any],
) -> str:
    """
    Support both Arabic field names:

    - text_ar
    - ar_text

    The current raw dataset uses ar_text.
    """

    arabic_text = article.get("text_ar")

    if arabic_text is None:
        arabic_text = article.get("ar_text")

    if arabic_text is None:
        return ""

    return str(arabic_text)


# ============================================================
# Single Article Preprocessing
# ============================================================


def preprocess_article(
    article: dict[str, Any],
) -> dict[str, Any]:
    """
    Preprocess one Civil Code article.

    The original Arabic text is preserved in text_ar.
    A normalized version is stored in text_ar_normalized.
    """

    if not isinstance(article, dict):
        raise TypeError(
            "Each article must be a dictionary"
        )

    # --------------------------------------------------------
    # Article number
    # --------------------------------------------------------

    if "article_number" not in article:
        raise KeyError(
            "Missing required field: article_number"
        )

    article_number = normalize_article_number(
        article["article_number"]
    )

    # --------------------------------------------------------
    # Arabic text
    # --------------------------------------------------------

    arabic_text = get_arabic_text(article)

    # Preserve original Arabic text
    original_arabic = arabic_text

    # Create normalized copy for search / embeddings
    normalized_arabic = normalize_arabic_text(
        arabic_text
    )

    # --------------------------------------------------------
    # English text
    # --------------------------------------------------------

    english_text = article.get(
        "text_en",
        "",
    )

    normalized_english = normalize_english_text(
        str(english_text)
    )

    # --------------------------------------------------------
    # Repealed status
    # --------------------------------------------------------

    original_repealed = article.get(
        "is_repealed"
    )

    if original_repealed is True:
        repealed = True
    else:
        repealed = is_repealed_article(
            article_number
        )

    # --------------------------------------------------------
    # Processed article
    # --------------------------------------------------------

    return {
        "article_number": article_number,
        "book": normalize_optional_text(
            article.get("book")
        ),
        "chapter": normalize_optional_text(
            article.get("chapter")
        ),
        "section": normalize_optional_text(
            article.get("section")
        ),
        "topic": normalize_optional_text(
            article.get("topic")
        ),
        "text_ar": original_arabic,
        "text_ar_normalized": normalized_arabic,
        "text_en": normalized_english,
        "is_repealed": repealed,
        "source_page": article.get("source_page"),
        "citation": normalize_optional_text(
            article.get("citation")
        ),
    }


# ============================================================
# Dataset Preprocessing
# ============================================================


def preprocess_dataset(
    input_path: str | Path,
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Read raw JSON, preprocess all articles,
    sort them by article number, and save the result.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Read JSON
    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Expected JSON file to contain a list of articles."
        )

    # Process articles
    processed_articles = [
        preprocess_article(article)
        for article in data
    ]

    # Sort by article number
    processed_articles.sort(
        key=lambda article: article[
            "article_number"
        ]
    )

    # Create output directory
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save processed JSON
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            processed_articles,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return processed_articles


# ============================================================
# Main
# ============================================================


def main() -> None:
    """
    Run the preprocessing pipeline using paths
    defined in config.py.
    """

    processed_articles = preprocess_dataset(
        input_path=RAW_JSON_PATH,
        output_path=PROCESSED_JSON_PATH,
    )

    print(
        f"Processed {len(processed_articles)} articles."
    )

    print(
        f"Saved to: {PROCESSED_JSON_PATH}"
    )


if __name__ == "__main__":
    main()
