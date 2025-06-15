from typing import Dict, List, Any, Optional
import json
import logging
import re
from dataclasses import asdict

from src.document_config import DocumentTypeConfig
from src.ocr.postprocess import normalize_ocr_lines

logger = logging.getLogger(__name__)

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

def create_extraction_prompt(
    normalized_lines: List[Dict[str, Any]],
    doc_config: DocumentTypeConfig
) -> str:
    """Create a prompt for the LLM to extract fields from the document."""
    
    # Format the document content
    doc_content = []
    for item in normalized_lines:
        if item["type"] == "label_value":
            doc_content.append(f"{item['label']}: {item['value']}")
        else:
            doc_content.append(item["text"])
    
    # Create the prompt
    prompt = f"""You are a document processing assistant specializing in credit request forms. Extract the following fields from the document below.

Document Type: {doc_config.name}

Required Fields:
{chr(10).join(f"- {field}: {doc_config.field_descriptions[field]}" for field in doc_config.expected_fields)}

Document Content:
{chr(10).join(doc_content)}

Please extract the fields and return them in the following JSON format:
{{
    "extracted_fields": {{
        "field_name": {{
            "value": "extracted value",
            "confidence": 0.95,  # confidence score between 0 and 1, must be a number
            "source": "label_value or text_line"  # where the value was found
        }}
    }},
    "missing_fields": ["list", "of", "fields", "that", "could", "not", "be", "extracted"]
}}

Important Guidelines:
1. For monetary values, extract only the number without currency symbols
2. For dates, use the format YYYY-MM-DD
3. For checkboxes, use "true" or "false"
4. For percentages, extract only the number without the % symbol
5. For areas, extract only the number without the unit
6. Only include fields that you are confident about
7. If you're not sure about a field, include it in missing_fields
8. Your response must be valid JSON. Do not include any additional text or explanations outside the JSON structure.
9. The 'source' field must be either 'label_value' or 'text_line', nothing else.
10. Do not wrap your response in markdown code blocks or add any explanatory text.
11. The 'confidence' field must be a number between 0 and 1, never null or undefined.

Example of monetary value extraction:
Input: "4.200.000€" -> Output: "4200000"
Input: "Ca. 18.000 €" -> Output: "18000"

Example of checkbox extraction:
Input: "[x] ja" -> Output: "true"
Input: "[ ] nein" -> Output: "false"
"""

    return prompt

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, handling common formatting issues."""
    # Try direct JSON parsing first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in markdown code blocks
    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    match = re.search(code_block_pattern, response)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON in the response without code blocks
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, response)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If all else fails, try to extract just the extracted_fields part
    fields_pattern = r'"extracted_fields"\s*:\s*(\{[\s\S]*?\})'
    match = re.search(fields_pattern, response)
    if match:
        try:
            fields_json = match.group(1)
            # Ensure it's a complete JSON object
            if fields_json.startswith('{') and fields_json.endswith('}'):
                return {
                    "extracted_fields": json.loads(fields_json),
                    "missing_fields": []
                }
        except json.JSONDecodeError:
            pass
    
    # If all else fails, return empty result
    logger.warning(f"Could not parse JSON from response: {response[:200]}...")
    return {
        "extracted_fields": {},
        "missing_fields": []
    }

def validate_extracted_fields(
    extracted_fields: Dict[str, Any],
    doc_config: DocumentTypeConfig
) -> Dict[str, Any]:
    """Validate extracted fields against the document configuration rules."""
    
    validation_results = {}
    
    for field_name, field_data in extracted_fields.items():
        if field_name not in doc_config.validation_rules:
            validation_results[field_name] = {
                "is_valid": True,
                "message": "No validation rules defined"
            }
            continue
            
        rules = doc_config.validation_rules[field_name]
        value = field_data["value"]
        
        # Type validation
        if rules.get("type") == "number":
            try:
                # Handle German number format (1.234,56)
                if isinstance(value, str):
                    value = value.replace(".", "").replace(",", ".")
                value = float(value)
                
                if "min" in rules and value < rules["min"]:
                    validation_results[field_name] = {
                        "is_valid": False,
                        "message": f"Value {value} is below minimum {rules['min']}"
                    }
                    continue
                    
                if "max" in rules and value > rules["max"]:
                    validation_results[field_name] = {
                        "is_valid": False,
                        "message": f"Value {value} is above maximum {rules['max']}"
                    }
                    continue
                    
            except ValueError:
                validation_results[field_name] = {
                    "is_valid": False,
                    "message": f"Value {value} is not a valid number"
                }
                continue
        
        # Pattern validation
        if "pattern" in rules:
            import re
            if not re.match(rules["pattern"], str(value)):
                validation_results[field_name] = {
                    "is_valid": False,
                    "message": f"Pattern validation failed: {rules.get('description', 'Invalid format')}"
                }
                continue
        
        validation_results[field_name] = {
            "is_valid": True,
            "message": "Validation passed"
        }
    
    return validation_results

async def extract_fields_with_llm(
    normalized_lines: List[Dict[str, Any]],
    doc_config: DocumentTypeConfig,
    llm_client: Any,  # Replace with your actual LLM client type
    original_ocr_lines: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Extract fields from document using LLM.
    
    Args:
        normalized_lines: List of normalized OCR lines
        doc_config: Document type configuration
        llm_client: LLM client instance
        original_ocr_lines: Optional list of original OCR lines with bounding boxes and confidence
    
    Returns:
        Dict containing extracted fields, validation results, and any errors
    """
    try:
        # Return empty result for empty input
        if not normalized_lines:
            return {
                "extracted_fields": {},
                "missing_fields": doc_config.expected_fields,
                "validation_results": {}
            }
            
        # Create the extraction prompt
        prompt = create_extraction_prompt(normalized_lines, doc_config)
        
        # Call the LLM
        response = await llm_client.generate(prompt)
        
        # Parse the response
        result = extract_json_from_response(response)
        
        # Validate extracted fields
        if "extracted_fields" in result:
            # Ensure source values are valid and confidence is numeric
            for field_name, field_data in result["extracted_fields"].items():
                if field_data["source"] not in ["label_value", "text_line"]:
                    field_data["source"] = "text_line"  # Default to text_line if invalid
                
                # Add original OCR data if available
                if original_ocr_lines:
                    ocr_data = find_original_ocr_data(field_data["value"], original_ocr_lines)
                    if ocr_data:
                        field_data.update(ocr_data)
                    else:
                        # If no OCR data found, set default confidence
                        field_data["confidence"] = 0.5
                else:
                    # If no original OCR lines provided, set default confidence
                    field_data["confidence"] = 0.5
            
            validation_results = validate_extracted_fields(
                result["extracted_fields"],
                doc_config
            )
            result["validation_results"] = validation_results
        
        return result
        
    except Exception as e:
        logger.exception("Error during field extraction")
        return {
            "error": str(e),
            "extracted_fields": {},
            "missing_fields": doc_config.expected_fields,
            "validation_results": {}
        } 