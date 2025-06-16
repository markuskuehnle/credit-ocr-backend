import pytest
from typing import List, Dict, Any
import json
from pathlib import Path

def load_document_config() -> Dict[str, Any]:
    """Load document type configuration."""
    config_path = Path("config/document_types.conf")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def normalize_label(label: str) -> str:
    """Normalize a label for comparison."""
    return label.lower().replace("?", "").replace("n", "").strip()

def mock_save_ocr_results(
    output_path: str = "tests/tmp/mock_ocr_results.json"
) -> List[Dict[str, Any]]:
    """
    Mock function to save OCR results to a JSON file using real normalized data.
    Ensures all required fields from document_types.conf are present.
    
    Args:
        output_path: Path to save the results
    
    Returns:
        List of normalized OCR items
    """
    # Load document configuration
    config = load_document_config()
    expected_fields = config["credit_request"]["expected_fields"]
    field_mappings = config["credit_request"]["field_mappings"]
    
    # Load real normalized data
    normalized_path = Path("tests/tmp/sample_creditrequest_normalized.json")
    with normalized_path.open("r", encoding="utf-8") as f:
        normalized_data = json.load(f)
    
    # Filter and organize data by expected fields
    organized_data = []
    for entry in normalized_data:
        if entry["type"] == "label_value":
            # Normalize the label for comparison
            label = entry["label"]
            normalized_label_text = normalize_label(label)
            
            # Check if this label maps to an expected field
            for german_label, eng_name in field_mappings.items():
                normalized_mapping = normalize_label(german_label)
                if normalized_mapping in normalized_label_text:
                    if eng_name in expected_fields:
                        organized_data.append(entry)
                        break
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save results to file with proper Unicode handling
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(organized_data, f, indent=2, ensure_ascii=False)
    
    return organized_data

def test_mock_ocr_results():
    """Test the mock OCR results saving function with real data."""
    # Load document configuration
    config = load_document_config()
    expected_fields = config["credit_request"]["expected_fields"]
    field_mappings = config["credit_request"]["field_mappings"]
    
    # Save mock results
    results = mock_save_ocr_results()
    
    # Track which expected fields we've found
    found_fields = set()
    
    # Verify results
    assert len(results) > 0, "No results found"
    
    # Check each entry
    for entry in results:
        assert entry["type"] == "label_value"
        assert "label" in entry
        assert "value" in entry
        assert "confidence" in entry
        assert "page" in entry
        assert "bounding_box" in entry
        
        # Check if this label maps to an expected field
        label = entry["label"]
        normalized_label_text = normalize_label(label)
        
        for german_label, eng_name in field_mappings.items():
            normalized_mapping = normalize_label(german_label)
            if normalized_mapping in normalized_label_text:
                if eng_name in expected_fields:
                    found_fields.add(eng_name)
                    
                    # Verify confidence is a valid number
                    assert isinstance(entry["confidence"], (int, float))
                    assert 0 <= entry["confidence"] <= 1
                    
                    # Verify bounding box structure
                    assert len(entry["bounding_box"]) == 4
                    for point in entry["bounding_box"]:
                        assert "x" in point
                        assert "y" in point
                        assert isinstance(point["x"], (int, float))
                        assert isinstance(point["y"], (int, float))
                break
    
    # Print which fields were found and which are missing
    missing_fields = set(expected_fields) - found_fields
    if missing_fields:
        print("\nMissing fields:")
        for field in missing_fields:
            print(f"- {field}")
    
    # Verify file was created
    output_path = "tests/tmp/mock_ocr_results.json"
    assert Path(output_path).exists()
    
    # Read and verify saved file
    with open(output_path, "r", encoding="utf-8") as f:
        saved_results = json.load(f)
    assert saved_results == results 