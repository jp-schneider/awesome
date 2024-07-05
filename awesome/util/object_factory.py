import inspect
from typing import Any, Type


class ObjectFactory:
    """Class for creating arbitrary objects from kwargs or dicts and set their properties."""

    FORBIDDEN_ATTRIBUTES = ["__class__"]

    @staticmethod
    def is_pydantic_type(_type: Type) -> bool:
        """Test whether the object is a pydantic type.

        Parameters
        ----------
        _type : Type
            The type to check.

        Returns
        -------
        bool
            Whether its a pydantic model type.
        """
        if _type is None:
            return False
        if _type is not None:
            if 'pydantic.main.BaseModel' in _type.__module__ + "." + _type.__name__:
                return True
            else:
                return ObjectFactory.is_pydantic_type(_type.__base__)

    @staticmethod
    def create_from_kwargs(_type: Type, allow_dynamic_args: bool = True, **kwargs) -> Any:
        """Factory method to create a instance of a type with arbitrary args.
        It will check the constructor for allowed arguments and passes all kwargs
        in the constructor which are matched by name. Other arguments will be set
        dynamically if allow_dynamic_args is set.

        Parameters
        ----------
        _type : Type
            The type to create.
        allow_dynamic_args : bool, optional
            If dynamic args should be allowed, by default True
        kwargs
            All properties which should be set in the object.
            
        Returns
        -------
        Any
            An object of type _type.
        """
    
        is_pydantic = ObjectFactory.is_pydantic_type(_type)

        # If the type is from pydantic the object is directly created for validation.
        if is_pydantic:
            return _type(**kwargs)

        # Getting init properties from constructor
        init_properties = list(inspect.signature(_type.__init__).parameters)[1:]

        # Split kwargs in class declared args and dynamic args.
        declared_args, dynamic_args = {}, {}
        for name, val in kwargs.items():
            if name in init_properties:
                declared_args[name] = val
            else:
                dynamic_args[name] = val

        obj = _type(**declared_args)

        if allow_dynamic_args:
            for property, value in dynamic_args.items():
                if property in ObjectFactory.FORBIDDEN_ATTRIBUTES:
                    continue
                setattr(obj, property, value)
        return obj

    @staticmethod
    def create_empty_and_fill(_type: Type, **kwargs) -> Any:
        """Creates an instance of _type without any args and sets all kwargs dynamically.

        Parameters
        ----------
        _type : Type
            The instance to create.

        kwargs
            All properties which should be set in the object.

        Returns
        -------
        Any
            The created object.
        """
        obj = _type()
        for new_name, new_val in kwargs.items():
            setattr(obj, new_name, new_val)
        return obj
