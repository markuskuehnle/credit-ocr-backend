import json
from pathlib import Path
from typing import Any

from src.ocr.azure_ocr_client import analyze_single_document_with_azure
from src.ocr.postprocess import extract_text_lines_with_bbox_and_confidence


def test_azure_ocr_runs_successfully() -> None:
    """Analyzes a test PDF and writes OCR output to JSON with text, polygon-based bounding box, confidence, and page info."""

    sample_pdf_path = Path("tests/tmp/sample.pdf")
    output_path = Path("tests/tmp/ocr_result.json")

    assert sample_pdf_path.exists(), f"Sample PDF not found at {sample_pdf_path}"

    # Run OCR
    result = analyze_single_document_with_azure(str(sample_pdf_path))
    assert result is not None, "Azure OCR returned None"
    assert hasattr(result, "pages") and result.pages, "OCR result has no pages"

    # Postprocess result
    ocr_data: list[dict[str, Any]] = extract_text_lines_with_bbox_and_confidence(result)

    # Basic checks
    assert isinstance(ocr_data, list), "OCR output is not a list"
    assert len(ocr_data) > 0, "No OCR entries extracted"

    for entry in ocr_data:
        assert isinstance(entry, dict), f"Non-dict entry: {entry}"
        assert "text" in entry and isinstance(entry["text"], str), f"Invalid text in: {entry}"
        assert "page" in entry and isinstance(entry["page"], int), f"Missing or invalid page in: {entry}"
        assert "confidence" in entry, f"Missing confidence in: {entry}"
        assert "bounding_box" in entry, f"Missing bounding_box in OCR entry: {entry}"

        if entry["bounding_box"] is not None:
            assert isinstance(entry["bounding_box"], list)
            assert all("x" in p and "y" in p for p in entry["bounding_box"])
        assert entry["type"] in ("line", "word"), f"Unexpected entry type: {entry}"

        if entry["confidence"] is not None:
            assert isinstance(entry["confidence"], float), f"Confidence is not a float in: {entry}"

    # Write to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(ocr_data, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(ocr_data)} OCR entries to {output_path}")
