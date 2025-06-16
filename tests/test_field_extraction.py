import json
import pytest
from pathlib import Path
from typing import Dict, Any
import logging
import asyncio

from src.llm.client import OllamaClient
from src.llm.field_extractor import extract_fields_with_llm, validate_field, load_document_config
from src.config import AppConfig, DocumentTypeConfig
from tests.environment.environment import app_config

logger = logging.getLogger(__name__)

# Load test configuration
app_config = AppConfig("tests/resources/test_application.conf")

@pytest.fixture
def llm_client():
    """Create a test LLM client."""
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
    """Fixture providing a sample credit request document configuration."""
    return DocumentTypeConfig(
        name="Kreditantrag",
        expected_fields=[
            "company_name",
            "legal_form",
            "founding_date",
            "business_address",
            "commercial_register",
            "vat_id",
            "website",
            "property_type",
            "property_name",
            "property_address",
            "purchase_price",
            "requested_amount",
            "purpose",
            "equity_share",
            "construction_year",
            "total_area",
            "loan_amount",
            "term",
            "monthly_payment",
            "interest_rate",
            "early_repayment",
            "public_funding"
        ],
        field_descriptions={
            "company_name": "Name of the company",
            "legal_form": "Legal form of the company (e.g., GmbH, AG)",
            "founding_date": "Date when the company was founded",
            "business_address": "Business address of the company",
            "commercial_register": "Commercial register number",
            "vat_id": "VAT identification number",
            "website": "Company website URL",
            "property_type": "Type of property (e.g., office, retail)",
            "property_name": "Name of the property",
            "property_address": "Address of the property",
            "purchase_price": "Purchase price of the property",
            "requested_amount": "Requested loan amount",
            "purpose": "Purpose of the loan",
            "equity_share": "Share of equity in the project",
            "construction_year": "Year the property was constructed",
            "total_area": "Total area of the property",
            "loan_amount": "Total loan amount",
            "term": "Loan term in years",
            "monthly_payment": "Monthly payment amount",
            "interest_rate": "Interest rate for the loan",
            "early_repayment": "Whether early repayment is allowed",
            "public_funding": "Whether public funding is available"
        },
        field_mappings={
            "Firmenname": "company_name",
            "Rechtsform": "legal_form",
            "Gründungsdatum": "founding_date",
            "Geschäftsadresse": "business_address",
            "Handelsregisternummer": "commercial_register",
            "USt-IdNr": "vat_id",
            "Webseite": "website",
            "Immobilienart": "property_type",
            "Objektbezeichnung": "property_name",
            "Objektadresse": "property_address",
            "Kaufpreis": "purchase_price",
            "Kreditsumme": "requested_amount",
            "Verwendungszweck": "purpose",
            "Eigenkapitalanteil": "equity_share",
            "Baujahr": "construction_year",
            "Gesamtfläche": "total_area",
            "Darlehenssumme": "loan_amount",
            "Laufzeit": "term",
            "Ratenwunsch": "monthly_payment",
            "Zinssatz": "interest_rate",
            "Sondertilgungen gewünscht": "early_repayment",
            "Öffentliche Fördermittel beantragt": "public_funding"
        },
        validation_rules={
            "company_name": {"type": "string", "required": True},
            "legal_form": {"type": "string", "required": True},
            "founding_date": {"type": "string", "required": True},
            "business_address": {"type": "string", "required": True},
            "commercial_register": {"type": "string", "required": True},
            "vat_id": {"type": "string", "required": True},
            "website": {"type": "string", "required": False},
            "property_type": {"type": "string", "required": True},
            "property_name": {"type": "string", "required": True},
            "property_address": {"type": "string", "required": True},
            "purchase_price": {"type": "number", "required": True},
            "requested_amount": {"type": "number", "required": True},
            "purpose": {"type": "string", "required": True},
            "equity_share": {"type": "number", "required": True},
            "construction_year": {"type": "number", "required": True},
            "total_area": {"type": "number", "required": True},
            "loan_amount": {"type": "number", "required": True},
            "term": {"type": "string", "required": True},
            "monthly_payment": {"type": "number", "required": True},
            "interest_rate": {"type": "number", "required": True},
            "early_repayment": {"type": "boolean", "required": True},
            "public_funding": {"type": "boolean", "required": True}
        }
    )

@pytest.fixture
def document_config():
    """Load document configuration for testing."""
    config_path = Path("config/document_types.conf")
    assert config_path.exists(), f"Configuration file not found: {config_path}"
    return load_document_config(config_path)

@pytest.fixture
def sample_ocr_lines():
    """Sample OCR lines for testing."""
    return [
        {"type": "line", "text": "Firmenname", "page": 1, "bounding_box": [{"x": 0.5, "y": 1.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "DemoTech GmbH", "page": 1, "bounding_box": [{"x": 3.0, "y": 1.01}]*4, "confidence": 0.98},
        {"type": "line", "text": "Rechtsform", "page": 1, "bounding_box": [{"x": 0.5, "y": 2.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "Gesellschaft mit beschränkter Haftung (GmbH)", "page": 1, "bounding_box": [{"x": 3.0, "y": 2.01}]*4, "confidence": 0.97},
        {"type": "line", "text": "Gründungsdatum", "page": 1, "bounding_box": [{"x": 0.5, "y": 3.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "15.03.2018", "page": 1, "bounding_box": [{"x": 3.0, "y": 3.01}]*4, "confidence": 0.99},
        {"type": "line", "text": "Geschäftsanschrift", "page": 1, "bounding_box": [{"x": 0.5, "y": 4.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "Hauptstraße 123, 70173 Stuttgart", "page": 1, "bounding_box": [{"x": 3.0, "y": 4.01}]*4, "confidence": 0.96},
        {"type": "line", "text": "Handelsregisternummer / Gericht", "page": 1, "bounding_box": [{"x": 0.5, "y": 5.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "HRB 123456 / Amtsgericht Stuttgart", "page": 1, "bounding_box": [{"x": 3.0, "y": 5.01}]*4, "confidence": 0.97},
        {"type": "line", "text": "USt-ID / Steuernummer", "page": 1, "bounding_box": [{"x": 0.5, "y": 6.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "DE123456789", "page": 1, "bounding_box": [{"x": 3.0, "y": 6.01}]*4, "confidence": 0.98},
        {"type": "line", "text": "Website (optional)", "page": 1, "bounding_box": [{"x": 0.5, "y": 7.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "www.demotech.de", "page": 1, "bounding_box": [{"x": 3.0, "y": 7.01}]*4, "confidence": 0.99},
        {"type": "line", "text": "Art der Immobilie", "page": 1, "bounding_box": [{"x": 0.5, "y": 8.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "Gewerbeimmobilie - Bürogebäude", "page": 1, "bounding_box": [{"x": 3.0, "y": 8.01}]*4, "confidence": 0.96},
        {"type": "line", "text": "Objektbezeichnung", "page": 1, "bounding_box": [{"x": 0.5, "y": 9.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "InnovationsCampus Stuttgart", "page": 1, "bounding_box": [{"x": 3.0, "y": 9.01}]*4, "confidence": 0.97},
        {"type": "line", "text": "Adresse", "page": 1, "bounding_box": [{"x": 0.5, "y": 10.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "Innovationsntraße 1, 70469 Stuttgart", "page": 1, "bounding_box": [{"x": 3.0, "y": 10.01}]*4, "confidence": 0.96},
        {"type": "line", "text": "Kaufpreis / Baukosten", "page": 1, "bounding_box": [{"x": 0.5, "y": 11.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "4.200.000€", "page": 1, "bounding_box": [{"x": 3.0, "y": 11.01}]*4, "confidence": 0.98},
        {"type": "line", "text": "Gewünschte Finanzierungssumme", "page": 1, "bounding_box": [{"x": 0.5, "y": 12.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "3.500.000€", "page": 1, "bounding_box": [{"x": 3.0, "y": 12.01}]*4, "confidence": 0.98},
        {"type": "line", "text": "Verwendungszweck", "page": 1, "bounding_box": [{"x": 0.5, "y": 13.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "Kauf und Renovierung", "page": 1, "bounding_box": [{"x": 3.0, "y": 13.01}]*4, "confidence": 0.96},
        {"type": "line", "text": "Eigenkapitalanteil", "page": 1, "bounding_box": [{"x": 0.5, "y": 14.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "700.000€", "page": 1, "bounding_box": [{"x": 3.0, "y": 14.01}]*4, "confidence": 0.98},
        {"type": "line", "text": "Baujahr", "page": 1, "bounding_box": [{"x": 0.5, "y": 15.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "1995", "page": 1, "bounding_box": [{"x": 3.0, "y": 15.01}]*4, "confidence": 0.99},
        {"type": "line", "text": "Fläche gesamt", "page": 1, "bounding_box": [{"x": 0.5, "y": 16.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "2.800 m²", "page": 1, "bounding_box": [{"x": 3.0, "y": 16.01}]*4, "confidence": 0.97},
        {"type": "line", "text": "Gewünschte Darlehenssumme", "page": 1, "bounding_box": [{"x": 0.5, "y": 17.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "3.500.000€", "page": 1, "bounding_box": [{"x": 3.0, "y": 17.01}]*4, "confidence": 0.98},
        {"type": "line", "text": "Laufzeit", "page": 1, "bounding_box": [{"x": 0.5, "y": 18.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "20 Jahre", "page": 1, "bounding_box": [{"x": 3.0, "y": 18.01}]*4, "confidence": 0.99},
        {"type": "line", "text": "Ratenwunsch", "page": 1, "bounding_box": [{"x": 0.5, "y": 19.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "Ca. 18.000 € (monatlich)", "page": 1, "bounding_box": [{"x": 3.0, "y": 19.01}]*4, "confidence": 0.96},
        {"type": "line", "text": "Zinssatz", "page": 1, "bounding_box": [{"x": 0.5, "y": 20.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "Festzins, 3.2% p.a.", "page": 1, "bounding_box": [{"x": 3.0, "y": 20.01}]*4, "confidence": 0.97},
        {"type": "line", "text": "Sondertilgungen gewünscht", "page": 1, "bounding_box": [{"x": 0.5, "y": 21.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "[x] ja [ ] nein", "page": 1, "bounding_box": [{"x": 3.0, "y": 21.01}]*4, "confidence": 0.98},
        {"type": "line", "text": "Öffentliche Fördermittel beantragt?", "page": 1, "bounding_box": [{"x": 0.5, "y": 22.0}]*4, "confidence": 0.95},
        {"type": "line", "text": "[ ] ja [x] nein", "page": 1, "bounding_box": [{"x": 3.0, "y": 22.01}]*4, "confidence": 0.98},
    ]

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
    
    validation_results = {}
    for field, value in test_fields.items():
        if field in credit_request_config.validation_rules:
            rules = credit_request_config.validation_rules[field]
            validation_results[field] = validate_field(value, rules)
    
    # Check validation results
    assert isinstance(validation_results, dict)
    for field_name, field_validation in validation_results.items():
        assert isinstance(field_validation, dict)
        assert "is_valid" in field_validation
        assert isinstance(field_validation["is_valid"], bool)
        if not field_validation["is_valid"]:
            assert "errors" in field_validation
            assert isinstance(field_validation["errors"], list)

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
        ocr_lines=sample_normalized_result,
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
        # If field has OCR data, it should have bounding box and page
        if "bounding_box" in field_data:
            assert "page" in field_data
            assert isinstance(field_data["bounding_box"], list)
            assert isinstance(field_data["page"], int)
        
        # Check confidence value
        confidence = field_data["confidence"]
        if confidence is not None:
            assert isinstance(confidence, (int, float)), f"Confidence must be numeric, got {type(confidence)}"
            assert 0 <= confidence <= 1, f"Confidence must be between 0 and 1, got {confidence}"
    
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
        ocr_lines=[],
        doc_config=credit_request_config,
        llm_client=llm_client
    )
    
    assert isinstance(result, dict)
    assert "extracted_fields" in result
    assert len(result["extracted_fields"]) == 0
    assert "missing_fields" in result
    assert len(result["missing_fields"]) == len(credit_request_config.expected_fields)
    assert "validation_results" in result
    assert len(result["validation_results"]) == 0

@pytest.mark.asyncio
async def test_extract_fields_from_sample(sample_ocr_lines, document_config, llm_client):
    """Test field extraction from sample OCR lines."""
    result = await extract_fields_with_llm(
        ocr_lines=sample_ocr_lines,
        doc_config=document_config["credit_request"],
        llm_client=llm_client
    )

    # Check that we got some fields
    assert len(result["extracted_fields"]) > 0

    # Check that all fields have the expected structure
    for field_name, field_data in result["extracted_fields"].items():
        assert isinstance(field_data, dict)
        assert "value" in field_data
        assert "confidence" in field_data
        # If field has OCR data, it should have bounding box and page
        if "bounding_box" in field_data:
            assert "page" in field_data
            assert isinstance(field_data["bounding_box"], list)
            assert isinstance(field_data["page"], int)

@pytest.mark.asyncio
async def test_extract_fields_from_real_ocr(document_config, llm_client):
    """Test field extraction from real OCR results."""
    ocr_result_path = Path("tests/tmp/sample_creditrequest_ocr_result.json")
    assert ocr_result_path.exists(), f"Test file not found: {ocr_result_path}"

    with ocr_result_path.open("r", encoding="utf-8") as f:
        ocr_lines = json.load(f)

    result = await extract_fields_with_llm(
        ocr_lines=ocr_lines,
        doc_config=document_config["credit_request"],
        llm_client=llm_client
    )

    # Check that we got some fields
    assert len(result["extracted_fields"]) > 0

    # Check that all fields have the expected structure
    for field_name, field_data in result["extracted_fields"].items():
        assert isinstance(field_data, dict)
        assert "value" in field_data
        assert "confidence" in field_data
        # If field has OCR data, it should have bounding box and page
        if "bounding_box" in field_data:
            assert "page" in field_data
            assert isinstance(field_data["bounding_box"], list)
            assert isinstance(field_data["page"], int)

    # Save output for inspection
    out_path = Path("tests/tmp/sample_creditrequest_extracted_fields.json")
    with out_path.open("w", encoding="utf-8") as out_f:
        json.dump(result, out_f, ensure_ascii=False, indent=2) 