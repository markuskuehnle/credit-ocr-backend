from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from pdf2image import convert_from_path
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import fitz
import shutil

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
    extracted_fields: Dict[str, Any],
    output_path: Path,
    ocr_lines: List[Dict[str, Any]] = None
) -> None:
    """
    Visualize extracted fields on the PDF by drawing bounding boxes and confidence scores.
    
    Args:
        pdf_path: Path to the input PDF file
        extracted_fields: Dictionary of extracted fields with their values and metadata
        output_path: Path where the visualization should be saved
        ocr_lines: Optional list of original OCR lines for reference
    """
    # Convert PDF to images with higher DPI for better quality
    images = convert_from_path(str(pdf_path), dpi=150)
    logger.info(f"Converted PDF to {len(images)} images")

    # Process each page
    for page_num, image in enumerate(images, 1):
        # Create drawing context
        draw = ImageDraw.Draw(image)
        
        # Try to load Arial font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial", 16)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw bounding boxes for each field
        boxes_drawn = 0
        for field_name, field_data in extracted_fields.items():
            if not isinstance(field_data, dict):
                continue
                
            value = field_data.get("value")
            if value is None:
                continue

            # Find matching OCR line for this field value
            matching_line = None
            if ocr_lines:
                for line in ocr_lines:
                    if line["text"] == str(value) and line.get("page") == page_num:
                        matching_line = line
                        break

            if matching_line and matching_line.get("bounding_box"):
                # Get bounding box coordinates
                bbox = matching_line["bounding_box"]
                confidence = matching_line.get("confidence", 0.5)
                
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
                text = f"{field_name}: {value} ({confidence:.2f})"
                text_width = font.getlength(text)
                text_x = points[0][0]  # Left edge of box
                text_y = min(p[1] for p in points) - 25  # 25 pixels above highest point of box
                
                # Draw text with black outline for better visibility
                for offset in [(1,1), (-1,-1), (1,-1), (-1,1)]:
                    draw.text((text_x + offset[0], text_y + offset[1]), text, fill=(0,0,0), font=font)
                draw.text((text_x, text_y), text, fill=color, font=font)
                boxes_drawn += 1
            else:
                # If no matching OCR line found, log it
                logger.warning(f"No OCR line found for field {field_name} with value {value} on page {page_num}")

        logger.info(f"Drew {boxes_drawn} bounding boxes on page {page_num}")

        # Save visualization (save all pages, even if no boxes were drawn)
        output_file = output_path.parent / f"{output_path.stem}_page{page_num}.png"
        image.save(output_file)
        logger.info(f"Created visualization at {output_file}") 