from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import fitz
import shutil
from collections import defaultdict
from src.config import DocumentTypeConfig, DocumentProcessingConfig

logger = logging.getLogger(__name__)

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_confidence_color(confidence: float | None) -> str:
    """Get color based on confidence score."""
    if confidence is None:
        return "#808080"  # Gray for unknown confidence
    if confidence >= 0.8:
        return "#00FF00"  # Green for high confidence
    elif confidence >= 0.6:
        return "#FFFF00"  # Yellow for medium confidence
    else:
        return "#FF0000"  # Red for low confidence

def draw_bounding_box(
    draw: ImageDraw.ImageDraw,
    bbox: List[Dict[str, float]],
    color: str,
    label: str,
    confidence: float | None,
    page_width: int,
    page_height: int
) -> None:
    """Draw a bounding box with label and confidence score."""
    try:
        # Convert inch-based coordinates to pixel coordinates
        # Assuming standard 72 DPI for PDF and coordinates are in inches
        DPI = 72
        points = []
        for point in bbox:
            # Convert inches to pixels and scale to page dimensions
            x = int((point["x"] * DPI) * (page_width / (8.5 * DPI)))  # Assuming 8.5" width
            y = int((point["y"] * DPI) * (page_height / (11 * DPI)))  # Assuming 11" height
            points.append((x, y))
        
        # Draw the bounding box
        draw.polygon(points, outline=color, width=3)
        
        # Draw label with background
        label_text = f"{label} ({confidence:.2f})" if confidence is not None else label
        try:
            font = ImageFont.truetype("Arial", 20)
        except IOError:
            font = ImageFont.load_default()
        
        # Calculate text size
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw text background
        text_x = points[0][0]
        text_y = points[0][1] - text_height - 5
        draw.rectangle(
            [(text_x, text_y), (text_x + text_width, text_y + text_height)],
            fill="white"
        )
        
        # Draw text
        draw.text((text_x, text_y), label_text, fill=color, font=font)
        
        logger.debug(f"Drew bounding box for {label} at {points} with color {color}")
        
    except Exception as e:
        logger.error(f"Error drawing bounding box for {label}: {str(e)}")

def visualize_extracted_fields(
    pdf_path: Path,
    normalized_data: List[Dict[str, Any]],
    output_path: Path,
    doc_config: DocumentTypeConfig = None
) -> None:
    """
    Visualize extracted fields on PDF pages using normalized data.
    
    Args:
        pdf_path: Path to the PDF file
        normalized_data: List of normalized OCR items (label_value pairs and text lines)
        output_path: Path where to save the output images
        doc_config: Document type configuration containing field mappings
    """
    if not normalized_data:
        logger.warning("No normalized data provided")
        return

    if not doc_config:
        # Load document configuration if not provided
        try:
            doc_config = DocumentProcessingConfig.from_json("config/document_types.conf")
            doc_config = doc_config.document_types["credit_request"]
        except Exception as e:
            logger.warning(f"Failed to load document configuration: {e}")
            return

    # Convert PDF pages to images
    images = convert_from_path(pdf_path, dpi=150)

    # Group normalized items by page for faster lookup
    items_by_page = defaultdict(list)
    for item in normalized_data:
        if item.get("bounding_box"):
            items_by_page[item["page"]].append(item)

    # Load font for text
    try:
        font = ImageFont.truetype("Arial", 12)
    except IOError:
        font = ImageFont.load_default()

    # Process each page
    for page_num, image in enumerate(images, 1):
        draw = ImageDraw.Draw(image)
        boxes_drawn = 0

        # Get items for this page
        page_items = items_by_page.get(page_num, [])

        # Process each label-value pair
        for item in page_items:
            # Get the canonical field name from the label
            field_name = None
            label_text = item.get("label", item.get("text", ""))
            normalized_label = label_text.lower().replace("?", "").replace("n", "").strip()
            
            for german_label, eng_name in doc_config.field_mappings.items():
                normalized_mapping = german_label.lower().replace("?", "").replace("n", "").strip()
                if normalized_mapping in normalized_label:
                    field_name = eng_name
                    break

            if not field_name:
                continue

            # Get bounding box and confidence
            bbox = item.get("bounding_box")
            confidence = item.get("confidence", 0.5)
            
            if not bbox:
                continue

            # Scale coordinates from inches to pixels (150 DPI)
            points = [(int(p["x"] * 150), int(p["y"] * 150)) for p in bbox]
            
            # Choose color based on confidence
            if confidence >= 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence >= 0.6:
                color = (255, 165, 0)  # Orange for medium confidence
            else:
                color = (255, 0, 0)  # Red for low confidence
            
            # Draw bounding box with thicker line
            draw.polygon(points, outline=color, width=3)
            
            # Calculate text position (directly above the box)
            value = item.get("value", item.get("text", ""))
            text = f"{field_name}: {value} ({confidence:.2f})"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_height = text_bbox[3] - text_bbox[1]
            text_x = points[0][0]  # Left edge of box
            text_y = min(p[1] for p in points) - text_height
            
            # Draw text with black outline for better visibility
            for offset in [(1,1), (-1,-1), (1,-1), (-1,1)]:
                draw.text((text_x + offset[0], text_y + offset[1]), text, fill=(0,0,0), font=font)
            draw.text((text_x, text_y), text, fill=color, font=font)
            
            boxes_drawn += 1

        # Save the page image
        output_file = output_path.parent / f"{output_path.stem}_page{page_num}.png"
        image.save(output_file, "PNG")
        logger.info(f"Drew {boxes_drawn} boxes on page {page_num}") 