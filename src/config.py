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
class Database:
    host: str
    port: int
    name: str
    user: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class Redis:
    host: str
    port: int
    db: int
    password: str
    broker_url: str
    result_backend: str
    task_serializer: str
    accept_content: List[str]
    result_serializer: str
    timezone: str
    enable_utc: bool


@dataclass
class AzureStorage:
    connection_string: str
    account_name: str
    account_key: str
    endpoint: str
    container_name: str


@dataclass
class AzureFormRecognizer:
    endpoint: str
    key: str


@dataclass
class Azure:
    storage: AzureStorage
    form_recognizer: AzureFormRecognizer


@dataclass
class AppSettings:
    debug: bool
    log_level: str
    timezone: str


@dataclass
class OllamaConfig:
    image: str
    exposed_port: int
    bind_port_host: int
    bind_port_container: int
    container_name: str
    cache_dir: str
    mem_limit: str
    host: str
    port: int

    @classmethod
    def from_config_file(cls, config_path: str) -> 'OllamaConfig':
        """Load Ollama configuration from HOCON file."""
        try:
            config_data = ConfigFactory.parse_file(config_path)
            ollama_config = config_data.get('ollama', {})
            
            return cls(
                image=ollama_config.get('image'),
                exposed_port=ollama_config.get('exposed_port'),
                bind_port_host=ollama_config.get('bind_port_host'),
                bind_port_container=ollama_config.get('bind_port_container'),
                container_name=ollama_config.get('container_name'),
                cache_dir=ollama_config.get('cache_dir'),
                mem_limit=ollama_config.get('mem_limit'),
                host=ollama_config.get('host'),
                port=ollama_config.get('port')
            )
        except Exception as e:
            logger.error(f"Failed to load Ollama configuration from {config_path}: {e}")
            raise


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
    database: Database
    redis: Redis
    azure: Azure
    app: AppSettings
    document_config: Optional[DocumentProcessingConfig] = None

    def __init__(self, config_dir: str = "config"):
        """Initialize configuration from multiple HOCON files."""
        try:
            config_dir_path = Path(config_dir)
            
            # Load application configuration
            app_config_path = config_dir_path / "application.conf"
            if app_config_path.exists():
                app_config_data = ConfigFactory.parse_file(str(app_config_path))
            else:
                logger.warning(f"Application configuration file not found at {app_config_path}")
                app_config_data = ConfigTree()
            
            # Load database configuration
            db_config_path = config_dir_path / "database.conf"
            if db_config_path.exists():
                db_config_data = ConfigFactory.parse_file(str(db_config_path))
            else:
                logger.warning(f"Database configuration file not found at {db_config_path}")
                db_config_data = ConfigTree()
            
            # Load Redis configuration
            redis_config_path = config_dir_path / "redis.conf"
            if redis_config_path.exists():
                redis_config_data = ConfigFactory.parse_file(str(redis_config_path))
            else:
                logger.warning(f"Redis configuration file not found at {redis_config_path}")
                redis_config_data = ConfigTree()
            
            # Load Azure configuration
            azure_config_path = config_dir_path / "azure.conf"
            if azure_config_path.exists():
                azure_config_data = ConfigFactory.parse_file(str(azure_config_path))
            else:
                logger.warning(f"Azure configuration file not found at {azure_config_path}")
                azure_config_data = ConfigTree()
            
            # Load LLM configuration
            llm_config = app_config_data.get('generative_llm', {})
            
            # Check for environment variable overrides for Ollama
            ollama_host = os.environ.get('OLLAMA_HOST')
            ollama_port = os.environ.get('OLLAMA_PORT')
            
            if ollama_host and ollama_port:
                # Use environment variables if available
                llm_url = f"http://{ollama_host}:{ollama_port}"
                logger.info(f"Using Ollama URL from environment variables: {llm_url}")
            else:
                # Use config file URL
                llm_url = llm_config.get('url')
                logger.info(f"Using Ollama URL from config file: {llm_url}")
            
            self.generative_llm = GenerativeLlm(
                url=llm_url,
                model_name=llm_config.get('model_name')
            )
            
            # Load database configuration
            db_config = db_config_data.get('database', {})
            self.database = Database(
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 5432),
                name=db_config.get('name', 'dms_meta'),
                user=db_config.get('user', 'dms'),
                password=db_config.get('password', 'dms'),
                pool_size=db_config.get('pool_size', 10),
                max_overflow=db_config.get('max_overflow', 20),
                pool_timeout=db_config.get('pool_timeout', 30),
                pool_recycle=db_config.get('pool_recycle', 3600)
            )
            
            # Load Redis configuration
            redis_config = redis_config_data.get('redis', {})
            celery_config = redis_config.get('celery', {})
            self.redis = Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                password=redis_config.get('password', ''),
                broker_url=celery_config.get('broker_url', 'redis://localhost:6379/0'),
                result_backend=celery_config.get('result_backend', 'redis://localhost:6379/0'),
                task_serializer=celery_config.get('task_serializer', 'json'),
                accept_content=celery_config.get('accept_content', ['json']),
                result_serializer=celery_config.get('result_serializer', 'json'),
                timezone=celery_config.get('timezone', 'Europe/Berlin'),
                enable_utc=celery_config.get('enable_utc', True)
            )
            
            # Load Azure configuration
            azure_config = azure_config_data.get('azure', {})
            storage_config = azure_config.get('storage', {})
            form_recognizer_config = azure_config.get('form_recognizer', {})
            
            self.azure = Azure(
                storage=AzureStorage(
                    connection_string=storage_config.get('connection_string'),
                    account_name=storage_config.get('account_name'),
                    account_key=storage_config.get('account_key'),
                    endpoint=storage_config.get('endpoint'),
                    container_name=storage_config.get('container_name')
                ),
                form_recognizer=AzureFormRecognizer(
                    endpoint=form_recognizer_config.get('endpoint'),
                    key=form_recognizer_config.get('key')
                )
            )
            
            # Load app settings
            app_settings = app_config_data.get('app', {})
            self.app = AppSettings(
                debug=app_settings.get('debug', False),
                log_level=app_settings.get('log_level', 'INFO'),
                timezone=app_settings.get('timezone', 'Europe/Berlin')
            )
            
            # Load document configuration from config/document_types.conf
            doc_config_path = config_dir_path / 'document_types.conf'
            if doc_config_path.exists():
                try:
                    self.document_config = DocumentProcessingConfig.from_json(str(doc_config_path))
                except Exception as e:
                    logger.warning(f"Failed to load document configuration from {doc_config_path}: {e}")
                    self.document_config = None
            else:
                logger.warning(f"Document configuration file not found at {doc_config_path}")
                self.document_config = None
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_dir}: {e}")
            raise
