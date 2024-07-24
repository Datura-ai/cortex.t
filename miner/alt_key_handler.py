from os import environ, path


from msgspec import yaml
from sys import argv


def check_endpoint_overrides() -> bool:
    """Check if the endpoint_overrides.yaml file exists in the current directory or script directory.

    Returns:
        bool: True if the file exists in either location, False otherwise."""

    file_path = path.join(path.dirname(__file__), "endpoint_overrides.yaml")
    script_dir_file_path = path.join(path.dirname(path.abspath(argv[0])), "endpoint_overrides.yaml")
    return path.exists(file_path) or path.exists(script_dir_file_path)


def get_endpoint_overrides() -> dict:
    """Get endpoint overrides from the endpoint_overrides.yaml file.

    Returns:
        dict: A dictionary containing the endpoint overrides.

    Raises:
        FileNotFoundError: If the endpoint_overrides.yaml file is not found.
        Exception: If an error occurs while reading the file.
    """

    if not check_endpoint_overrides():
        return {}
    try:
        with open(path.join(path.dirname(__file__), "endpoint_overrides.yaml"), "r") as f:
            return yaml.decode(f.read())
    except FileNotFoundError:
        with open(path.join(path.dirname(path.abspath(argv[0])), "endpoint_overrides.yaml"), "r") as f:
            return yaml.decode(f.read())
    except Exception:
        raise


def override_endpoint_keys() -> None:
    """Override endpoint keys based on the values provided in the endpoint_overrides.yaml file.

    Args:
        None

    Returns:
        None"""

    environ.update(get_endpoint_overrides().get("OVERRIDE_ENVIRONMENT_KEYS", {}))
