from typing import Any
from azure.ai.formrecognizer import AnalyzeResult


def extract_text_lines_with_bbox_and_confidence(result: AnalyzeResult) -> list[dict[str, Any]]:
    """Extracts lines and words with text, bounding box, confidence, and page number from Azure OCR results."""

    extracted: list[dict[str, Any]] = []

    for page in result.pages:
        page_number = page.page_number

        # Extract lines (no confidence available for lines)
        for line in page.lines:
            extracted.append({
                "type": "line",
                "text": line.content,
                "page": page_number,
                "bounding_box": [{"x": p.x, "y": p.y} for p in line.polygon] if line.polygon else None,
                "confidence": None,
            })

        # Extract words (has confidence)
        for word in page.words:
            extracted.append({
                "type": "word",
                "text": word.content,
                "page": page_number,
                "bounding_box": [{"x": p.x, "y": p.y} for p in word.polygon] if word.polygon else None,
                "confidence": round(word.confidence, 2) if word.confidence else None,
            })

    return extracted
