class ProviderRegistryMeta(type):
    _registry = {}

    def __new__(cls, name, bases, attrs):
        # Create the new class
        new_class = super().__new__(cls, name, bases, attrs)
        # Register the class name and the class itself
        if bases:
            # Register the class
            cls._registry[name] = new_class

        return new_class

    @classmethod
    def get_class(cls, name):
        # Retrieve the class by name from the registry
        return cls._registry.get(name)

    @classmethod
    def all_classes(cls):
        # Return all registered classes
        return cls._registry


class ServiceRegistryMeta(type):
    _registry = {}

    def __new__(cls, name, bases, attrs):
        # Create the new class
        new_class = super().__new__(cls, name, bases, attrs)
        # Register the class name and the class itself
        if bases:
            cls._registry[name] = new_class
        return new_class

    @classmethod
    def get_class(cls, name):
        # Retrieve the class by name from the registry
        return cls._registry.get(name)

    @classmethod
    def all_classes(cls):
        # Return all registered classes
        return cls._registry


class ValidatorRegistryMeta(type):
    _registry = {}

    def __new__(cls, name, bases, attrs):
        # Create the new class
        new_class = super().__new__(cls, name, bases, attrs)
        # Register the class name and the class itself
        if bases:
            cls._registry[name] = new_class
        return new_class

    @classmethod
    def get_class(cls, name):
        # Retrieve the class by name from the registry
        return cls._registry.get(name)

    @classmethod
    def all_classes(cls):
        # Return all registered classes
        return cls._registry
