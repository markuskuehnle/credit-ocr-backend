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

        # Extract lines with confidence from their words
        for line in page.lines:
            # Calculate line confidence as average of its words' confidence
            word_confidences = []
            for word in page.words:
                # Check if word is part of this line by comparing bounding boxes
                if word.polygon and line.polygon:
                    # Simple overlap check - if word's center is within line's box
                    word_center_x = sum(p.x for p in word.polygon) / len(word.polygon)
                    word_center_y = sum(p.y for p in word.polygon) / len(word.polygon)
                    line_min_x = min(p.x for p in line.polygon)
                    line_max_x = max(p.x for p in line.polygon)
                    line_min_y = min(p.y for p in line.polygon)
                    line_max_y = max(p.y for p in line.polygon)
                    
                    if (line_min_x <= word_center_x <= line_max_x and 
                        line_min_y <= word_center_y <= line_max_y):
                        if word.confidence is not None:
                            word_confidences.append(word.confidence)
            
            # Calculate average confidence for the line
            line_confidence = None
            if word_confidences:
                line_confidence = round(sum(word_confidences) / len(word_confidences), 2)
            
            extracted.append({
                "type": "line",
                "text": line.content,
                "page": page_number,
                "bounding_box": [{"x": p.x, "y": p.y} for p in line.polygon] if line.polygon else None,
                "confidence": line_confidence,
            })

        # Extract words with their confidence
        for word in page.words:
            extracted.append({
                "type": "word",
                "text": word.content,
                "page": page_number,
                "bounding_box": [{"x": p.x, "y": p.y} for p in word.polygon] if word.polygon else None,
                "confidence": round(word.confidence, 2) if word.confidence is not None else None,
            })

    return extracted


def extract_label_value_pairs(ocr_lines: List[Dict[str, Any]], y_thresh=0.2, x_split=2.5) -> List[Dict[str, str]]:
    """
    Extract label-value pairs from OCR lines using flexible heuristics:
    - Handles same-line colon-separated labels
    - Handles label on one line and value on the next line (layout-aware)
    - Handles table-like structures with aligned labels and values
    - Uses bounding box center positions for x/y analysis
    - Works per page
    """

    def get_center_y(box):  # average Y of bounding box
        return sum(p["y"] for p in box) / len(box) if box else 0.0

    def get_center_x(box):
        return sum(p["x"] for p in box) / len(box) if box else 0.0

    def get_box_width(box):
        if not box:
            return 0.0
        return max(p["x"] for p in box) - min(p["x"] for p in box)

    def get_box_height(box):
        if not box:
            return 0.0
        return max(p["y"] for p in box) - min(p["y"] for p in box)

    def are_horizontally_aligned(box1, box2, threshold=0.1):
        """Check if two boxes are roughly on the same horizontal line"""
        if not box1 or not box2:
            return False
        y1 = get_center_y(box1)
        y2 = get_center_y(box2)
        # Use box height to determine alignment threshold
        height = max(get_box_height(box1), get_box_height(box2))
        return abs(y1 - y2) <= height * threshold

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
        
        # First pass: handle colon-separated pairs
        for i, line in enumerate(lines):
            if i in used_indices:
                continue
                
            text = line["text"].strip()
            if ":" in text:
                label, value = map(str.strip, text.split(":", 1))
                if label and value:
                    results.append({
                        "label": label,
                        "value": value,
                        "page": page,
                        "confidence": line.get("confidence"),
                        "bounding_box": line.get("bounding_box")
                    })
                    used_indices.add(i)

        # Second pass: handle table-like structures and adjacent lines
        for i, line in enumerate(lines):
            if i in used_indices:
                continue

            text = line["text"].strip()
            cx = get_center_x(line["bounding_box"])
            cy = get_center_y(line["bounding_box"])

            # Check if this line is likely a label (left-aligned)
            if cx < x_split:
                # Look for value in horizontally aligned lines
                best_value_line = None
                best_value_x = float('inf')
                
                for j, next_line in enumerate(lines):
                    if j in used_indices:
                        continue
                        
                    next_cx = get_center_x(next_line["bounding_box"])
                    
                    # Check if lines are horizontally aligned and the next line is to the right
                    if (are_horizontally_aligned(line["bounding_box"], next_line["bounding_box"]) and 
                        next_cx > cx and next_cx < best_value_x):
                        best_value_line = next_line
                        best_value_x = next_cx
                
                if best_value_line:
                    # Found a horizontally aligned value
                    results.append({
                        "label": text,
                        "value": best_value_line["text"].strip(),
                        "page": page,
                        "confidence": min(line.get("confidence", 1.0), 
                                       best_value_line.get("confidence", 1.0)),
                        "bounding_box": best_value_line.get("bounding_box")
                    })
                    used_indices.update([i, lines.index(best_value_line)])
                    continue
                
                # If no horizontal match, look for value in next few lines
                for j in range(i + 1, min(i + 3, len(lines))):
                    if j in used_indices:
                        continue
                    next_line = lines[j]
                    next_cx = get_center_x(next_line["bounding_box"])
                    next_cy = get_center_y(next_line["bounding_box"])
                    
                    # Check if the next line is a value (right-aligned and close vertically)
                    if next_cx > x_split and abs(next_cy - cy) <= y_thresh:
                        results.append({
                            "label": text,
                            "value": next_line["text"].strip(),
                            "page": page,
                            "confidence": min(line.get("confidence", 1.0), 
                                           next_line.get("confidence", 1.0)),
                            "bounding_box": next_line.get("bounding_box")
                        })
                        used_indices.update([i, j])
                        break

    return results


def normalize_ocr_lines(ocr_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize OCR lines into generic structured items.
    Returns a list of items like:
    - {'type': 'label_value', 'label': ..., 'value': ..., 'page': ..., 'confidence': ...}
    - {'type': 'text_line', 'text': ..., 'page': ..., 'confidence': ...}
    """
    structured = []

    # Detect label-value pairs first
    pairs = extract_label_value_pairs(ocr_lines)

    # Add label-value pairs
    for p in pairs:
        # Find the original OCR data for both label and value
        label_ocr = next((line for line in ocr_lines if line["text"] == p["label"]), None)
        value_ocr = next((line for line in ocr_lines if line["text"] == p["value"]), None)
        
        # Use the lower confidence of the two if both exist
        confidence = p.get("confidence")  # Use confidence from extract_label_value_pairs if available
        if confidence is None and label_ocr and value_ocr:
            label_conf = label_ocr.get("confidence")
            value_conf = value_ocr.get("confidence")
            if label_conf is not None and value_conf is not None:
                confidence = min(label_conf, value_conf)
            elif label_conf is not None:
                confidence = label_conf
            elif value_conf is not None:
                confidence = value_conf
        
        structured.append({
            "type": "label_value",
            "label": p["label"],
            "value": p["value"],
            "page": p["page"],
            "confidence": confidence,
            "bounding_box": p.get("bounding_box")  # Include bounding box from pairs
        })

    # Add all lines as text lines
    for line in ocr_lines:
        if line["type"] != "line":
            continue
        if line.get("bounding_box") is None:
            continue
        structured.append({
            "type": "text_line",
            "text": line["text"].strip(),
            "page": line["page"],
            "confidence": line.get("confidence"),
            "bounding_box": line.get("bounding_box")
        })

    return structured
