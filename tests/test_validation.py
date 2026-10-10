
import json

import pytest

from legal_rag.config import (
    PROCESSED_JSON_PATH,
    REPEALED_ARTICLES,
)

# ============================================================
# Dataset Validation Checks
# ============================================================
#
# 1. Article numbers are contiguous with no unexplained gaps.
# 2. Every article has non-empty Arabic text (text_ar).
# 3. No article exceeds a sane maximum length.
#    A giant record may indicate a failed article split.
# 4. Repealed articles are correctly flagged as is_repealed=True.


@pytest.fixture
def articles():
    """
    Load the processed Civil Code dataset.
    """

    with PROCESSED_JSON_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# ============================================================
# Validation Tests
# ============================================================


def test_article_numbers_are_contiguous(articles):
    """
    Article numbers should be contiguous unless
    a gap is explicitly explained.

    Example:
        1, 2, 3, 4, 5 -> valid

    A missing article such as:
        1, 2, 3, 5
    should fail because article 4 is unexplained.
    """

    article_numbers = [
        article["article_number"]
        for article in articles
    ]

    article_numbers = sorted(article_numbers)

    expected_numbers = list(
        range(
            article_numbers[0],
            article_numbers[-1] + 1,
        )
    )

    assert article_numbers == expected_numbers, (
        "Article numbers contain unexplained gaps. "
        f"Found: {article_numbers}"
    )


def test_every_article_has_non_empty_arabic_text(articles):
    """
    Every article must contain non-empty Arabic text.
    """

    empty_articles = [
        article["article_number"]
        for article in articles
        if not isinstance(article.get("text_ar"), str)
        or not article["text_ar"].strip()
    ]

    assert not empty_articles, (
        "The following articles have empty text_ar: "
        f"{empty_articles}"
    )


def test_no_article_exceeds_sane_length(articles):
    """
    A giant article may indicate a failed PDF split.

    Maximum allowed Arabic text length:
        10,000 characters
    """

    MAX_TEXT_LENGTH = 10_000

    oversized_articles = [
        (
            article["article_number"],
            len(article["text_ar"]),
        )
        for article in articles
        if len(article.get("text_ar", "")) > MAX_TEXT_LENGTH
    ]

    assert not oversized_articles, (
        "The following articles exceed the maximum "
        f"allowed length of {MAX_TEXT_LENGTH} characters: "
        f"{oversized_articles}"
    )


def test_repealed_articles_are_flagged(articles):
    """
    Articles known to be repealed must have
    is_repealed=True.
    """

    incorrectly_flagged = [
        article["article_number"]
        for article in articles
        if article["article_number"] in REPEALED_ARTICLES
        and article.get("is_repealed") is not True
    ]

    assert not incorrectly_flagged, (
        "The following repealed articles are not "
        f"flagged correctly: {incorrectly_flagged}"
    )



