from typing import List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass

from azure.ai.formrecognizer import AnalyzeResult


@dataclass
class DocumentTypeConfig:
    name: str
    expected_fields: List[str]
    field_descriptions: Dict[str, str]
    validation_rules: Dict[str, Any]  # Optional validation rules per field

@dataclass
class DocumentProcessingConfig:
    document_types: Dict[str, DocumentTypeConfig]


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


def extract_label_value_pairs(ocr_lines: List[Dict[str, Any]], y_thresh=0.2, x_split=2.5) -> List[Dict[str, str]]:
    """
    Extract label-value pairs from OCR lines using flexible heuristics:
    - Handles same-line colon-separated labels
    - Handles label on one line and value on the next line (layout-aware)
    - Uses bounding box center positions for x/y analysis
    - Works per page
    """

    def get_center_y(box):  # average Y of bounding box
        return sum(p["y"] for p in box) / len(box) if box else 0.0

    def get_center_x(box):
        return sum(p["x"] for p in box) / len(box) if box else 0.0

    # Sort lines by page and vertical position
    sorted_lines = sorted(
        ocr_lines,
        key=lambda x: (x["page"], get_center_y(x["bounding_box"]))
    )

    results = []
    page_buffers = defaultdict(list)

    # Group lines by page
    for line in sorted_lines:
        if line["type"] != "line":
            continue  # Only consider lines
        page = line["page"]
        page_buffers[page].append(line)

    for page, lines in page_buffers.items():
        used_indices = set()
        for i, line in enumerate(lines):
            text = line["text"].strip()
            cx = get_center_x(line["bounding_box"])
            cy = get_center_y(line["bounding_box"])

            # --- Case 1: Same-line label:value split ---
            if ":" in text:
                label, value = map(str.strip, text.split(":", 1))
                if label and value:
                    results.append({
                        "label": label,
                        "value": value,
                        "page": page
                    })
                    used_indices.add(i)
                    continue

            # --- Case 2: Label + Value on separate lines ---
            if i in used_indices:
                continue

            if cx < x_split:
                # likely a label line
                label = text
                y1 = cy

                # scan next few lines for a value
                for j in range(i + 1, min(i + 5, len(lines))):
                    if j in used_indices:
                        continue
                    next_line = lines[j]
                    next_cx = get_center_x(next_line["bounding_box"])
                    next_cy = get_center_y(next_line["bounding_box"])
                    if abs(next_cy - y1) > y_thresh:
                        break
                    if next_cx > x_split:
                        value = next_line["text"].strip()
                        results.append({
                            "label": label,
                            "value": value,
                            "page": page
                        })
                        used_indices.update([i, j])
                        break

    return results


def normalize_ocr_lines(ocr_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize OCR lines into generic structured items.
    Returns a list of items like:
    - {'type': 'label_value', 'label': ..., 'value': ..., 'page': ...}
    - {'type': 'text_line', 'text': ..., 'page': ...}
    """
    structured = []
    seen_indices = set()

    # Detect label-value pairs first
    pairs = extract_label_value_pairs(ocr_lines)

    for p in pairs:
        structured.append({
            "type": "label_value",
            "label": p["label"],
            "value": p["value"],
            "page": p["page"]
        })
        # optionally track used lines here to avoid duplication

    # Now add all remaining lines as plain text
    for line in ocr_lines:
        if line["type"] != "line":
            continue
        if line.get("bounding_box") is None:
            continue
        structured.append({
            "type": "text_line",
            "text": line["text"].strip(),
            "page": line["page"]
        })

    return structured
