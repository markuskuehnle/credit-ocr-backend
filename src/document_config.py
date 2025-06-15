from dataclasses import dataclass
from typing import Dict, List, Any
from pyhocon import ConfigFactory

@dataclass
class DocumentTypeConfig:
    name: str
    expected_fields: List[str]
    field_descriptions: Dict[str, str]
    validation_rules: Dict[str, Any]  # Optional validation rules per field

@dataclass
class DocumentProcessingConfig:
    document_types: Dict[str, DocumentTypeConfig]

    @classmethod
    def from_hocon(cls, config_path: str) -> 'DocumentProcessingConfig':
        """Load document processing configuration from a HOCON file."""
        config = ConfigFactory.parse_file(config_path)
        
        document_types = {}
        for doc_type, doc_config in config.get_config("document_types").items():
            document_types[doc_type] = DocumentTypeConfig(
                name=doc_type,
                expected_fields=doc_config.get_list("expected_fields"),
                field_descriptions=doc_config.get_config("field_descriptions").as_plain_ordered_dict(),
                validation_rules=doc_config.get_config("validation_rules").as_plain_ordered_dict() if doc_config.get("validation_rules") else {}
            )
        
        return cls(document_types=document_types)
