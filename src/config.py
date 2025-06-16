from dataclasses import dataclass, fields, is_dataclass
from enum import EnumMeta
from pathlib import Path
import os
import json
from typing import Dict, Any, List, Optional
import logging

from pyhocon import ConfigFactory, ConfigTree

logger = logging.getLogger(__name__)


def typed_value_from_config_tree(hocon: ConfigTree, field_type: type, field_name: str):
    if field_type == bool:
        return hocon.get_bool(field_name)
    if field_type == str:
        return hocon.get_string(field_name)
    if field_type == int:
        return hocon.get_int(field_name)
    if field_type == float:
        return hocon.get_float(field_name)
    if is_dataclass(field_type):
        sub_config = hocon.get_config(field_name)
        return dataclass_from_config_tree(sub_config, field_type)
    if isinstance(field_type, EnumMeta):
        enum_name = hocon.get_string(field_name)
        return field_type.from_name(enum_name)
    raise ValueError(
        f"Cannot convert the field {field_name} to type {field_type}. This is either a failure "
        f"in the config-file or a missing feature!"
    )


def dataclass_from_config_tree(ct: ConfigTree, data_cls: dataclass):
    args = [
        typed_value_from_config_tree(ct, field.type, field.name)
        for field in fields(data_cls)
    ]
    return data_cls(*args)


@dataclass
class GenerativeLlm:
    url: str
    model_name: str


@dataclass
class Minio:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    bucket: str
    region: str


@dataclass
class DocumentTypeConfig:
    name: str
    expected_fields: List[str]
    field_descriptions: Dict[str, str]
    validation_rules: Dict[str, Any]
    field_mappings: Dict[str, str] = None


@dataclass
class DocumentProcessingConfig:
    document_types: Dict[str, DocumentTypeConfig]

    @classmethod
    def from_json(cls, config_path: str) -> 'DocumentProcessingConfig':
        """Load document configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                doc_config = json.load(f)
            
            document_types = {}
            for doc_type, config in doc_config.items():
                document_types[doc_type] = DocumentTypeConfig(
                    name=config['name'],
                    expected_fields=config['expected_fields'],
                    field_descriptions=config['field_descriptions'],
                    validation_rules=config['validation_rules'],
                    field_mappings=config.get('field_mappings', {})
                )
            
            return cls(document_types=document_types)
        except Exception as e:
            logger.error(f"Failed to load document configuration from {config_path}: {e}")
            raise


@dataclass
class AppConfig:
    generative_llm: GenerativeLlm
    minio: Minio
    document_config: Optional[DocumentProcessingConfig] = None

    def __init__(self, config_path: str):
        """Initialize configuration from HOCON file."""
        try:
            # Parse HOCON configuration
            config_data = ConfigFactory.parse_file(config_path)
            
            # Load LLM configuration
            llm_config = config_data.get('generative_llm', {})
            self.generative_llm = GenerativeLlm(
                url=llm_config.get('url'),
                model_name=llm_config.get('model_name')
            )
            
            # Load Minio configuration
            minio_config = config_data.get('minio', {})
            self.minio = Minio(
                endpoint=minio_config.get('endpoint', 'localhost:9000'),
                access_key=minio_config.get('access_key', 'minioadmin'),
                secret_key=minio_config.get('secret_key', 'minioadmin'),
                secure=bool(minio_config.get('secure', False)),
                bucket=minio_config.get('bucket', 'credit-ocr'),
                region=minio_config.get('region', 'us-east-1')
            )
            
            # Load document configuration from config/document_types.conf
            doc_config_path = 'config/document_types.conf'
            if os.path.exists(doc_config_path):
                try:
                    self.document_config = DocumentProcessingConfig.from_json(doc_config_path)
                except Exception as e:
                    logger.warning(f"Failed to load document configuration from {doc_config_path}: {e}")
                    self.document_config = None
            else:
                logger.warning(f"Document configuration file not found at {doc_config_path}")
                self.document_config = None
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
