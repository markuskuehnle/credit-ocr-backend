import json
import logging
import shutil
from pathlib import Path

import pytest

from src.visualization.pdf_visualizer import visualize_extracted_fields

logger = logging.getLogger(__name__)

@pytest.mark.order(4)  # Run after field extraction tests
def test_visualize_extracted_fields(tmp_path):
    """Test visualization of extracted fields."""
    # Skip if poppler is not installed
    if not shutil.which("pdftoppm"):
        pytest.skip("poppler not installed")

    # Use the PDF file from tmp directory
    pdf_path = Path("tests/tmp/sample_creditrequest.pdf")
    logger.info(f"Looking for PDF at: {pdf_path.absolute()}")
    assert pdf_path.exists(), f"PDF file not found at {pdf_path}"

    # Load normalized data
    normalized_path = Path("tests/tmp/sample_creditrequest_normalized.json")
    logger.info(f"Looking for normalized data at: {normalized_path.absolute()}")
    assert normalized_path.exists(), f"Normalized data file not found at {normalized_path}"

    with open(normalized_path, "r", encoding="utf-8") as f:
        normalized_data = json.load(f)

    # Log the normalized data for debugging
    logger.info("Found the following normalized items:")
    for item in normalized_data:
        if item["type"] == "label_value":
            confidence = item.get("confidence", 0.0)
            logger.info(f"  {item['label']}: {item['value']} (confidence: {confidence})")

    # Create visualization
    output_path = Path("tests/tmp/sample_creditrequest_visualization")
    visualize_extracted_fields(
        pdf_path=pdf_path,
        normalized_data=normalized_data,
        output_path=output_path
    )

    # Verify visualization files were created
    page_files = list(Path("tests/tmp").glob("sample_creditrequest_visualization_page*.png"))
    assert len(page_files) > 0, "No visualization files were created"
    
    # Verify each file exists and is not empty
    for page_file in page_files:
        assert page_file.exists(), f"Visualization file {page_file} does not exist"
        assert page_file.stat().st_size > 0, f"Visualization file {page_file} is empty"
        logger.info(f"Created visualization at {page_file}") 