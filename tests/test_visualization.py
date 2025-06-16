import json
from pathlib import Path
import pytest
import shutil
import subprocess
from src.visualization.pdf_visualizer import visualize_extracted_fields
import logging
import fitz

logger = logging.getLogger(__name__)

def is_poppler_installed() -> bool:
    """Check if poppler is installed."""
    try:
        subprocess.run(['pdftoppm', '-v'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

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

    # Load extracted fields
    extracted_fields_path = Path("tests/tmp/sample_creditrequest_extracted_fields.json")
    logger.info(f"Looking for extracted fields at: {extracted_fields_path.absolute()}")
    assert extracted_fields_path.exists(), f"Extracted fields file not found at {extracted_fields_path}"
    
    with open(extracted_fields_path, "r", encoding="utf-8") as f:
        extracted_fields = json.load(f)
    
    # Verify we have the expected keys
    assert "extracted_fields" in extracted_fields
    assert "original" in extracted_fields
    assert "original_ocr" in extracted_fields["original"]
    
    logger.info(f"Number of extracted fields: {len(extracted_fields['extracted_fields'])}")
    logger.info(f"Number of OCR lines: {len(extracted_fields['original']['original_ocr'])}")

    # Create visualization
    output_path = Path("tests/tmp/sample_creditrequest_visualization")
    visualize_extracted_fields(
        pdf_path=pdf_path,
        extracted_fields=extracted_fields["extracted_fields"],
        output_path=output_path,
        ocr_lines=extracted_fields["original"]["original_ocr"]
    )

    # Verify visualization files were created
    page_files = list(Path("tests/tmp").glob("sample_creditrequest_visualization_page*.png"))
    assert len(page_files) > 0, "No visualization files were created"
    
    # Verify each file exists and is not empty
    for page_file in page_files:
        assert page_file.exists(), f"Visualization file {page_file} does not exist"
        assert page_file.stat().st_size > 0, f"Visualization file {page_file} is empty"
        logger.info(f"Created visualization at {page_file}") 