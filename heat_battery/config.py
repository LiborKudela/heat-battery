from .utilities import load_yaml
import os

this_file_dir = os.path.dirname(os.path.abspath(__file__))
GLOBAL_CONFIG_FILE_PATH = os.path.join(this_file_dir, "base_config.yaml")
LOCAL_CONFIG_FILE_PATH = os.path.join(os.getcwd(), "config.yaml")
MAX_CONFIG_DEPTH = 10

def setup_global_config_path(config_path: str) -> None:
    global GLOBAL_CONFIG_FILE_PATH
    if not os.path.exists(config_path):
        raise ValueError(f"Global config file not found at {config_path}")
    GLOBAL_CONFIG_FILE_PATH = config_path

def setup_local_config_path(config_path: str) -> None:
    global LOCAL_CONFIG_FILE_PATH
    if not os.path.exists(config_path):
        raise ValueError(f"Local config file not found at {config_path}")
    LOCAL_CONFIG_FILE_PATH = config_path

def merge_configs(
        base_config:dict, 
        update_config:dict, 
        current_path:list=[]
    ) -> dict:
    f"""
    Merges global/base and local configuration files. Local configuration takes
    precedence over global/base configuration. It also checks for any invalid
    paths in the update config file against the base config. A ValueError will be
    raised for the first detected invalid key that is not found in the
    global/base config.
    
    The function is recursive so max config depth is limited by hardcoded
    variable MAX_CONFIG_DEPTH = {MAX_CONFIG_DEPTH}. If the max depth is exceeded
    a ValueError is raised.
    
    Args:
        base_config (dict): Dictionary to update.
        update_config (dict): Dictionary with values to updated.
        current_path (list, optional): DO NOT USE THIS ARGUMENT it is internal. 

    Returns:
        dict: Merged configuration.
    """
    # TODO: This is a hack to avoid modifying the base_config in place.
    # if len(current_path) == 0:
    #     base_config = base_config.copy()
    assert len(current_path) <= MAX_CONFIG_DEPTH, "Config file is too deep"
    for key, value in update_config.items():
        if not key in base_config:
            raise ValueError(
                f"Update config file contains invalid key: {key}\n"
                f"Invalid config path: {' -> '.join(current_path)} -> {key}\n"
                f"Valid keys are: {list(base_config.keys())}\n"
            )
        if isinstance(base_config[key], dict):
            current_path.append(key)
            merge_configs(base_config[key], value, current_path)
            current_path.pop()
        else:
            base_config[key] = value
    return base_config

def get_current_config() -> dict:
    """
    Loads the global configuration file and merges it with local configuration
    where the local configuration takes precedence. Local configuration file has
    to be placed in the current working directory.

    Returns:
        dict: Merged configuration.
    """
    
    global_config = load_yaml(GLOBAL_CONFIG_FILE_PATH)
    if os.path.exists(LOCAL_CONFIG_FILE_PATH):
        local_config = load_yaml(LOCAL_CONFIG_FILE_PATH)
        global_config = merge_configs(global_config, local_config)
    return global_config

def assert_config_feature_enabled(feature_path: list[str], error_msg: str) -> None:
    """
    Asserts that a configuration feature is enabled, if not raises a ValueError.

    Args:
        feature_path (list[str]): Path to the feature in the configuration file.
    """
    value = get_current_config()
    for key in feature_path:
        value = value[key]
    assert isinstance(value, bool), (
        f"Expected a boolean value for feature {feature_path[-1]}, "
        f"got {type(value)}"
    )
    if not value:
        raise ValueError(error_msg)
    
def assert_config_value_set(value_path: list[str], error_msg: str) -> None:
    """
    Asserts that a configuration value is set, if not raises a ValueError.

    Args:
        value_path (list[str]): Path to the value in the configuration file.
        error_msg (str): Error message to raise if the value is not set.
    """
    value = get_current_config()
    for key in value_path:
        value = value[key]
    if value is None:   
        raise ValueError(f"{error_msg}")
    
def get_config_item(
        value_path: list[str],
        must_be_terminating_key: bool=True):
    """
    Gets a value from the configuration file. When the value is not set or the
    path is invalid a ValueError is raised. Some values need to be set in local
    configuration file such as database credentials to not be exposed. These localy
    unset values will raise a ValueError.
    
    Args:
        value_path (list[str]): Path to the value in the configuration file.
        must_be_terminating_key (bool, optional): If True, the function will raise
            a ValueError if the returned object is not a terminating key. Defaults
            to True.

    Returns:
        Any: Value or remaining dictionary at the path of config.
    """

    value = get_current_config()
    for i, key in enumerate(value_path):
        if i < len(value_path) - 1 and not isinstance(value[key], dict):
            raise ValueError(
                f"Config route ends with a non-dictionary value at path :'{value_path[:i]}' "
                "Please check your config file."
            )
        
        if key not in value:
            raise ValueError(
                f"Key: '{key}' does not exist for value path: {value_path[:i]}! "
                "Please check your config file."
            )   
        value = value[key]

    if value is None:
        raise ValueError(
            f"Key: '{key}' is not set for value path: {value_path[:i]}! "
            "Please check your config file."
        )

    if must_be_terminating_key and isinstance(value, dict):
        raise ValueError(
            f"Value for key: '{key}' is a non-terminating at path: {value_path[:i]}! "
            "Please check your config file."
        )

    return value