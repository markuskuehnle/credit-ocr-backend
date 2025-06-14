from dataclasses import dataclass, fields, is_dataclass
from enum import EnumMeta

from pyhocon import ConfigFactory, ConfigTree


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
    access_key: str
    secret_key: str
    endpoint: str


@dataclass
class AppConfig:
    indexing_state_filename: str
    generative_llm: GenerativeLlm
    minio: Minio = None

    def __init__(self, hocon_file_path=None):
        hocon = ConfigFactory.parse_file(hocon_file_path)

        for field_obj in fields(AppConfig):
            field_name = field_obj.name
            field_type = field_obj.type

            if is_dataclass(field_type):
                if hocon.get(field_name, None) is not None:
                    try:
                        config_subtree = hocon.get_config(field_name)
                        setattr(
                            self,
                            field_name,
                            dataclass_from_config_tree(config_subtree, field_type),
                        )
                    except Exception as e:
                        print(f"Failed to parse config section '{field_name}': {e}")
                        setattr(self, field_name, None)
                else:
                    setattr(self, field_name, None)
            else:
                if hocon.get(field_name, None) is not None:
                    try:
                        setattr(
                            self,
                            field_name,
                            typed_value_from_config_tree(hocon, field_type, field_name),
                        )
                    except Exception as e:
                        print(f"Failed to parse field '{field_name}': {e}")
                        setattr(self, field_name, None)
