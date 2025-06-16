import json
from pathlib import Path
import pytest
from src.ocr.postprocess import normalize_ocr_lines


@pytest.mark.order(2)  # Run second, after OCR test
def test_normalize_ocr_lines_from_sample() -> None:
    sample_lines = [
        {"type": "line", "text": "Firmenname", "page": 1, "bounding_box": [{"x": 0.5, "y": 1.0}]*4},
        {"type": "line", "text": "Demo Tech GmbH", "page": 1, "bounding_box": [{"x": 3.0, "y": 1.01}]*4},
        {"type": "line", "text": "USt-ID: DE123456789", "page": 1, "bounding_box": [{"x": 0.5, "y": 2.0}]*4},
    ]

    normalized = normalize_ocr_lines(sample_lines)

    label_value = [entry for entry in normalized if entry["type"] == "label_value"]
    text_lines = [entry for entry in normalized if entry["type"] == "text_line"]

    # Check for expected label-value pairs, ignoring confidence
    expected_pairs = [
        {"type": "label_value", "label": "Firmenname", "value": "Demo Tech GmbH", "page": 1},
        {"type": "label_value", "label": "USt-ID", "value": "DE123456789", "page": 1}
    ]
    
    for expected in expected_pairs:
        found = False
        for actual in label_value:
            if all(actual[k] == expected[k] for k in expected.keys()):
                found = True
                break
        assert found, f"Expected pair not found: {expected}"

    assert any("Demo Tech GmbH" in line["text"] for line in text_lines)


@pytest.mark.order(2)  # Run second, after OCR test
def test_normalize_ocr_lines_from_real_ocr() -> None:
    path = Path("tests/tmp/sample_creditrequest_ocr_result.json")
    assert path.exists(), f"Test file not found: {path}"

    with path.open("r", encoding="utf-8") as f:
        ocr_lines = json.load(f)

    normalized = normalize_ocr_lines(ocr_lines)

    label_value = [entry for entry in normalized if entry["type"] == "label_value"]
    text_lines = [entry for entry in normalized if entry["type"] == "text_line"]

    # Basic checks
    assert isinstance(normalized, list)
    assert all("type" in entry for entry in normalized)
    assert all(entry["type"] in {"label_value", "text_line"} for entry in normalized)

    # Targeted label/value existence check
    expected = {
        "type": "label_value",
        "label": "Rechtsform",
        "value": "Gesellschaft mit beschränkter Haftung (GmbH)",
        "page": 1,
    }
    
    # Check for expected pair, ignoring confidence
    found = False
    for actual in label_value:
        if all(actual[k] == expected[k] for k in expected.keys()):
            found = True
            break
    assert found, f"Expected pair not found: {expected}"

    # Ensure fallback preserved unstructured content
    assert any("Innovationsntraße" in line["text"] for line in text_lines), "Expected line text not found"

    # Save output for inspection
    out_path = Path("tests/tmp/sample_creditrequest_normalized.json")
    with out_path.open("w", encoding="utf-8") as out_f:
        json.dump(normalized, out_f, ensure_ascii=False, indent=2)