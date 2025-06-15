import json
import pytest
from pathlib import Path
from typing import Dict, Any

from src.llm.client import OllamaClient
from src.llm.field_extractor import extract_fields_with_llm, validate_extracted_fields
from src.document_config import DocumentTypeConfig
from src.config import AppConfig

# Load test configuration
app_config = AppConfig("tests/resources/test_application.conf")

@pytest.fixture
def llm_client():
    """Create an LLM client for testing."""
    return OllamaClient(
        base_url=app_config.generative_llm.url,
        model_name=app_config.generative_llm.model_name
    )

@pytest.fixture
def sample_ocr_result():
    """Load the sample OCR result from previous test."""
    path = Path("tests/tmp/sample_creditrequest_ocr_result.json")
    assert path.exists(), f"Test file not found: {path}"
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def sample_normalized_result():
    """Load the normalized OCR result from previous test."""
    path = Path("tests/tmp/sample_creditrequest_normalized.json")
    assert path.exists(), f"Test file not found: {path}"
    
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def credit_request_config():
    """Create a test document type configuration."""
    return DocumentTypeConfig(
        name="credit_request",
        expected_fields=[
            "company_name",
            "vat_id",
            "credit_amount",
            "purpose",
            "request_date"
        ],
        field_descriptions={
            "company_name": "The legal name of the company requesting credit",
            "vat_id": "The VAT identification number of the company",
            "credit_amount": "The requested credit amount in EUR",
            "purpose": "The purpose of the credit request",
            "request_date": "The date when the credit was requested"
        },
        validation_rules={
            "vat_id": {
                "pattern": r"^[A-Z]{2}[0-9A-Z]{8,12}$",
                "description": "VAT ID must start with 2 letters followed by 8-12 alphanumeric characters"
            },
            "credit_amount": {
                "type": "number",
                "min": 0,
                "description": "Credit amount must be a positive number"
            }
        }
    )

@pytest.mark.order(3)  # Run last, after postprocessing test
def test_validate_extracted_fields(credit_request_config):
    """Test field validation functionality."""
    test_fields = {
        "vat_id": {
            "value": "DE123456789",
            "confidence": 0.95,
            "source": "label_value"
        },
        "credit_amount": {
            "value": "1000.50",
            "confidence": 0.98,
            "source": "label_value"
        },
        "vat_id": {  # Test invalid VAT with same field name
            "value": "12345",  # Invalid VAT format
            "confidence": 0.90,
            "source": "text_line"
        },
        "credit_amount": {  # Test invalid amount with same field name
            "value": "-100",  # Negative amount
            "confidence": 0.95,
            "source": "label_value"
        }
    }
    
    validation_results = validate_extracted_fields(test_fields, credit_request_config)
    
    # Check validation results
    assert validation_results["vat_id"]["is_valid"] is False  # Should be invalid due to pattern
    assert validation_results["credit_amount"]["is_valid"] is False  # Should be invalid due to negative value
    
    # Check error messages
    assert "pattern" in validation_results["vat_id"]["message"].lower()
    assert "minimum" in validation_results["credit_amount"]["message"].lower()

@pytest.mark.order(3)  # Run last, after postprocessing test
@pytest.mark.asyncio
async def test_field_extraction_with_llm(
    llm_client,
    sample_normalized_result,
    sample_ocr_result,
    credit_request_config
):
    """Test the complete field extraction process with LLM."""
    result = await extract_fields_with_llm(
        normalized_lines=sample_normalized_result,
        doc_config=credit_request_config,
        llm_client=llm_client,
        original_ocr_lines=sample_ocr_result
    )

    # Basic structure checks
    assert isinstance(result, dict)
    assert "extracted_fields" in result
    assert "validation_results" in result

    # Check extracted fields
    extracted = result["extracted_fields"]
    assert isinstance(extracted, dict)

    # Check that we got some fields
    assert len(extracted) > 0

    # Check field structure
    for field_name, field_data in extracted.items():
        assert isinstance(field_data, dict)
        assert "value" in field_data
        assert "confidence" in field_data
        assert "source" in field_data
        
        # Check confidence value
        confidence = field_data["confidence"]
        if confidence is not None:
            assert isinstance(confidence, (int, float)), f"Confidence must be numeric, got {type(confidence)}"
            assert 0 <= confidence <= 1, f"Confidence must be between 0 and 1, got {confidence}"
        
        # Check source value
        assert field_data["source"] in ["label_value", "text_line"], \
            f"Source must be 'label_value' or 'text_line', got {field_data['source']}"

    # Check validation results
    validation = result["validation_results"]
    assert isinstance(validation, dict)
    for field_name, field_validation in validation.items():
        assert isinstance(field_validation, dict)
        assert "is_valid" in field_validation
        assert isinstance(field_validation["is_valid"], bool)
        if not field_validation["is_valid"]:
            assert "errors" in field_validation
            assert isinstance(field_validation["errors"], list)

    # Save results for inspection
    output_file = Path("tests/tmp/sample_creditrequest_extracted_fields.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

@pytest.mark.order(3)  # Run last, after postprocessing test
@pytest.mark.asyncio
async def test_field_extraction_error_handling(
    llm_client,
    credit_request_config
):
    """Test error handling in field extraction."""
    # Test with empty input
    result = await extract_fields_with_llm(
        normalized_lines=[],
        doc_config=credit_request_config,
        llm_client=llm_client
    )
    
    assert isinstance(result, dict)
    assert "extracted_fields" in result
    assert len(result["extracted_fields"]) == 0
    assert "missing_fields" in result
    assert len(result["missing_fields"]) == len(credit_request_config.expected_fields) 