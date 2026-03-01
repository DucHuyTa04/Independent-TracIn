import importlib


def load_class(class_path: str):
    """Load a class from full path string like module.submodule.ClassName."""
    if "." not in class_path:
        raise ValueError(f"Invalid class path: {class_path}")

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)

    if not hasattr(module, class_name):
        raise ValueError(f"Class {class_name} not found in module {module_name}")

    return getattr(module, class_name)
