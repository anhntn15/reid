import importlib


def class_importing(name):
    """
    helper function to use different class instance from config file
    import a class from a string name
    :param name: full path (with module specified) to class e.g: torch.nn.Linear
    :return: class instance
    """
    module_name, class_name = name.rsplit(".", 1)
    MyClass = getattr(importlib.import_module(module_name), class_name)
    return MyClass
