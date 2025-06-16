from typing import Dict, List, Any, Optional
import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from datetime import datetime

from src.llm.client import OllamaClient
from src.config import AppConfig, DocumentTypeConfig

logger = logging.getLogger(__name__)

def load_document_config(config_path: str) -> Dict[str, DocumentTypeConfig]:
    """Load document configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)

    document_types = {}
    for doc_type, doc_config in config_data.items():
        document_types[doc_type] = DocumentTypeConfig(
            name=doc_config['name'],
            expected_fields=doc_config['expected_fields'],
            field_descriptions=doc_config['field_descriptions'],
            validation_rules=doc_config['validation_rules'],
            field_mappings=doc_config.get('field_mappings', {})
        )

    return document_types

def find_original_ocr_data(text: str, ocr_lines: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the original OCR data for a given text."""
    for line in ocr_lines:
        if line["text"] == text:
            return {
                "bounding_box": line["bounding_box"],
                "confidence": line["confidence"],
                "page": line["page"]
            }
    return None

def clean_value(value: str, field_type: str) -> Any:
    """Clean and convert value based on field type."""
    if not value:
        return None

    if field_type == "string":
        return value.strip()
    
    elif field_type == "date":
        # Ensure date format DD.MM.YYYY
        if re.match(r"^\d{2}\.\d{2}\.\d{4}$", value):
            return value
        return None
    
    elif field_type == "currency":
        # Remove currency symbols, spaces, and convert comma to dot
        cleaned = value.replace("€", "").replace(" ", "").replace(",", ".")
        # Remove any non-numeric characters except decimal point
        cleaned = ''.join(c for c in cleaned if c.isdigit() or c == '.')
        return float(cleaned) if cleaned else None
    
    elif field_type == "area":
        # Remove unit and spaces
        cleaned = value.replace("m²", "").replace(" ", "")
        return float(cleaned) if cleaned else None
    
    elif field_type == "number":
        # Remove any non-numeric characters
        cleaned = ''.join(c for c in value if c.isdigit())
        return int(cleaned) if cleaned else None
    
    elif field_type == "boolean":
        return "[x]" in value.lower()
    
    return value

def extract_fields_with_llm(ocr_lines: List[Dict[str, Any]], document_type: str = "credit_request") -> Dict[str, Any]:
    """
    Extract fields from OCR lines using configuration-based rules.
    Returns a dictionary of field names to their values.
    """
    # Load document configuration
    config_path = Path("config/document_types.conf")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_document_config(config_path)
    if document_type not in config:
        raise ValueError(f"Unknown document type: {document_type}")
    
    # Extract fields from OCR lines
    extracted_fields = {}
    field_config = config[f"{document_type}.fields"]
    
    # Map OCR lines to fields
    for line in ocr_lines:
        if line["type"] != "line":
            continue

        text = line["text"].strip()
        confidence = line.get("confidence", 0.5)

        # Check each field's label in the configuration
        for field_name, field_rules in field_config.items():
            label = field_rules.get("label")
            if label and label in text:
                # Extract value by removing the label
                value = text.replace(label, "").strip()
                # Clean and convert value based on field type
                field_type = field_rules.get("type", "string")
                cleaned_value = clean_value(value, field_type)
                if cleaned_value is not None:
                    extracted_fields[field_name] = cleaned_value
                break

    return extracted_fields

def process_document(ocr_result_path: Path, document_type: str = "credit_request") -> Dict[str, Any]:
    """
    Process a document by extracting fields from OCR results.
    Returns a dictionary of field names to their values.
    """
    # Load OCR results
    with ocr_result_path.open("r", encoding="utf-8") as f:
        ocr_lines = json.load(f)

    # Extract fields
    extracted_fields = extract_fields_with_llm(ocr_lines, document_type)

    # Save extracted fields
    output_path = ocr_result_path.parent / f"{ocr_result_path.stem}_extracted_fields.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(extracted_fields, f, ensure_ascii=False, indent=2)

    return extracted_fields

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling potential text prefixes and comments."""
    try:
        # Find JSON between code blocks if present
        if "```" in response:
            # Find the first code block
            start = response.find("```")
            if start != -1:
                # Skip the opening ```
                start = response.find("\n", start) + 1
                # Find the closing ```
                end = response.find("```", start)
                if end != -1:
                    response = response[start:end].strip()
        
        # Remove any comments
        lines = []
        for line in response.split('\n'):
            if '//' in line:
                line = line[:line.find('//')]
            lines.append(line)
        response = '\n'.join(lines)
        
        # Try to parse the JSON
        return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from response: {e}")
        logger.error(f"Raw response: {response}")
        raise ValueError(f"Invalid JSON in response: {e}")

def create_extraction_prompt(ocr_lines: List[Dict[str, Any]], config: DocumentTypeConfig) -> str:
    """Create a prompt for field extraction."""
    # Create field descriptions with both English and German names
    field_descriptions = []
    for field, desc in config.field_descriptions.items():
        # Extract German name from description if available
        german_name = desc.split("(")[-1].strip(")") if "(" in desc else ""
        field_descriptions.append(f"- {field} ({german_name}): {desc}")

    # Create a mapping section to show exact field names
    field_mappings = []
    for german_name, english_name in config.field_mappings.items():
        field_mappings.append(f"- {german_name} → {english_name}")

    # Format OCR lines based on their type
    formatted_lines = []
    for line in ocr_lines:
        if line["type"] == "label_value":
            formatted_lines.append(f"{line['label']}: {line['value']}")
        elif line["type"] == "text_line":
            formatted_lines.append(line["text"])
        elif line["type"] == "line":
            formatted_lines.append(line["text"])

    # Construct the prompt
    prompt = f"""Extract the following fields from the document content below. Return a valid JSON object with the extracted fields.

Field Descriptions:
{chr(10).join(field_descriptions)}

Field Mappings (use these exact field names in your response):
{chr(10).join(field_mappings)}

Document Content:
{chr(10).join(formatted_lines)}

Instructions:
1. Return a valid JSON object with the extracted fields
2. Use the exact field names from the mappings above
3. Include only fields that are present in the document
4. For fields with units (e.g., years, currency), include the unit in the value
5. For boolean fields, return true/false
6. For dates, use the format DD.MM.YYYY
7. For numbers, include any units or currency symbols

Example response format:
{{
    "extracted_fields": {{
        "company_name": "Demo Tech GmbH",
        "legal_form": "GmbH",
        "founding_date": "01.01.2020",
        "business_address": "Musterstraße 123, 12345 Berlin",
        "purchase_price": "500.000 €",
        "term": "20 Jahre",
        "interest_rate": "3,5%"
    }},
    "missing_fields": ["website", "vat_id"],
    "validation_results": {{
        "company_name": {{"valid": true}},
        "legal_form": {{"valid": true}},
        "founding_date": {{"valid": true}}
    }}
}}

Please extract the fields from the document content above and return a JSON object in this format."""
    return prompt

def validate_field(value: Any, rules: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a field value against validation rules."""
    validation_result = {
        "is_valid": True,
        "errors": []
    }
    
    if not isinstance(value, dict) or "value" not in value:
        validation_result["is_valid"] = False
        validation_result["errors"].append("Invalid field format")
        return validation_result
    
    field_value = value["value"]
    
    # Type validation
    if "type" in rules:
        expected_type = rules["type"]
        if expected_type == "number":
            try:
                # Handle German number format (1.234,56)
                if isinstance(field_value, str):
                    field_value = field_value.replace(".", "").replace(",", ".")
                float(field_value)
            except (ValueError, TypeError):
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Value must be a number")
        elif expected_type == "boolean":
            if str(field_value).lower() not in ["true", "false"]:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Value must be a boolean")
        elif expected_type == "date":
            # Skip number validation for dates
            pass
    
    # Range validation (only for numbers)
    if "min" in rules and "type" in rules and rules["type"] == "number":
        try:
            if isinstance(field_value, str):
                field_value = field_value.replace(".", "").replace(",", ".")
            if float(field_value) < rules["min"]:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Value must be at least {rules['min']}")
        except (ValueError, TypeError):
            pass
    
    if "max" in rules and "type" in rules and rules["type"] == "number":
        try:
            if isinstance(field_value, str):
                field_value = field_value.replace(".", "").replace(",", ".")
            if float(field_value) > rules["max"]:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Value must be at most {rules['max']}")
        except (ValueError, TypeError):
            pass
    
    # Pattern validation
    if "pattern" in rules:
        import re
        if not re.match(rules["pattern"], str(field_value)):
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Value does not match required pattern")
    
    return validation_result

def validate_extracted_fields(fields: Dict[str, Any], doc_config: DocumentTypeConfig) -> Dict[str, Any]:
    """Validate all extracted fields against their validation rules."""
    validation_results = {}
    for field_name, field_data in fields.items():
        if field_name in doc_config.validation_rules:
            validation_results[field_name] = validate_field(field_data, doc_config.validation_rules[field_name])
    return validation_results

async def extract_fields_with_llm(
    ocr_lines: List[Dict[str, Any]],
    doc_config: DocumentTypeConfig,
    llm_client: OllamaClient,
    original_ocr_lines: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract fields from OCR lines using LLM.
    The LLM is only used to map OCR text to field names.
    Original OCR data (value, confidence, bounding box, page) is preserved.
    
    Args:
        ocr_lines: List of OCR lines with text and metadata
        doc_config: Document type configuration
        llm_client: LLM client for field extraction
        original_ocr_lines: Optional list of original OCR lines for reference
        
    Returns:
        Dictionary containing extracted fields, missing fields, and validation results
    """
    if not ocr_lines:
        return {
            "extracted_fields": {},
            "missing_fields": list(doc_config.expected_fields),
            "validation_results": {}
        }
        
    # Step 1: Let LLM map OCR text to field names
    prompt = create_extraction_prompt(ocr_lines, doc_config)
    response = await llm_client.generate(prompt)
    
    try:
        llm_result = extract_json_from_response(response)
    except ValueError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        logger.error(f"Raw response: {response}")
        raise
        
    # Step 2: Process extracted fields
    extracted_fields = {}
    for field_name, field_data in llm_result.get("extracted_fields", {}).items():
        # Ensure field data is a dictionary
        if not isinstance(field_data, dict):
            field_data = {"value": field_data}
            
        # Ensure required keys exist
        if "value" not in field_data:
            field_data["value"] = None
            
        # Step 3: Find matching normalized label-value pair
        if field_data["value"] is not None:
            value_str = str(field_data["value"]).lower()
            
            # Get all possible German labels for this field
            german_labels = []
            for label, eng_name in doc_config.field_mappings.items():
                if eng_name == field_name:
                    german_labels.append(label.lower())
            
            # First try to find a matching label-value pair
            matching_pair = None
            for line in ocr_lines:
                if line["type"] == "label_value":
                    line_label = line["label"].lower()
                    line_value = line["value"].lower()
                    
                    # Match if either the label or value matches
                    if (any(label in line_label for label in german_labels) or 
                        value_str in line_value):
                        matching_pair = line
                        break
            
            if matching_pair:
                # Use the label-value pair's data directly
                extracted_fields[field_name] = {
                    "value": matching_pair["value"],
                    "confidence": matching_pair.get("confidence", 0.5),
                    "bounding_box": matching_pair.get("bounding_box"),
                    "page": matching_pair.get("page")
                }
            else:
                # If no matching pair found, try to find matching OCR line
                matching_line = None
                if original_ocr_lines:
                    for line in original_ocr_lines:
                        if line["type"] != "line":
                            continue
                            
                        line_text = line["text"].lower()
                        
                        # Match if line contains either the value or any of the field's labels
                        if value_str in line_text or any(label in line_text for label in german_labels):
                            matching_line = line
                            break
                
                if matching_line:
                    # Use the OCR line's data directly
                    extracted_fields[field_name] = {
                        "value": matching_line["text"],
                        "confidence": matching_line.get("confidence", 0.5),
                        "bounding_box": matching_line.get("bounding_box"),
                        "page": matching_line.get("page")
                    }
                else:
                    # If no matching line found, use LLM output with default confidence
                    extracted_fields[field_name] = {
                        "value": field_data["value"],
                        "confidence": 0.5
                    }
        else:
            # If no value provided, use LLM output with default confidence
            extracted_fields[field_name] = {
                "value": field_data["value"],
                "confidence": 0.5
            }
            
    # Step 4: Apply field mappings
    mapped_fields = {}
    for field_name, field_data in extracted_fields.items():
        if field_name in doc_config.field_mappings:
            mapped_name = doc_config.field_mappings[field_name]
            mapped_fields[mapped_name] = field_data
        else:
            mapped_fields[field_name] = field_data
            
    # Step 5: Validate fields
    validation_results = validate_extracted_fields(mapped_fields, doc_config)
    
    # Prepare final result
    result = {
        "extracted_fields": mapped_fields,
        "missing_fields": llm_result.get("missing_fields", []),
        "validation_results": validation_results
    }
    
    return result 