from typing import Any, Dict, List, Literal, Optional, Set, Type
from abc import ABC, abstractmethod
from uuid import UUID

class JsonSerializationRule(ABC):
    """This is a base class for a runtime checkable json serialization rule. It is used to convert a value 
    based on its type to a different representation for storing."""

    priority: int 
    """Priority to apply this rule. Smaller priority means its checked earlier for applicability. Default is 100."""

    def __init__(self, priority: int = 100) -> None:
        super().__init__()
        self.priority = priority
    
    @classmethod
    @abstractmethod
    def applicable_forward_types(self) -> List[Type]:
        """Returns the types where this serialization rule is applicable for the forward path.

        Returns
        -------
        List[Type]
            Types where the rule can be applied.
        """
        ...

    def is_forward_applicable(self, value: Any) -> bool:
        """Testing whether a rules is applicable agains a given value.
        Can be overriden.

        Parameters
        ----------
        value : Any
            Value to test for applicability

        Returns
        -------
        bool
            If the type us applicable.
        """
        return isinstance(value, tuple(self.applicable_forward_types()))

    def is_forward_applicable(self, value: Any) -> bool:
        """Testing whether a rules is applicable agains a given value.
        Can be overriden.

        Parameters
        ----------
        value : Any
            Value to test for applicability

        Returns
        -------
        bool
            If the type us applicable.
        """
        return isinstance(value, tuple(self.applicable_forward_types()))

    @classmethod
    @abstractmethod
    def applicable_backward_types(self) -> List[Type]:
        """Returns the types where this serialization rule is applicable 
        for the backward path.

        Returns
        -------
        List[Type]
            Types where the rule can be applied.
        """
        ...

    def is_backward_applicable(self, value: Any) -> bool:
        """Testing whether a rules is applicable agains a given value.
        Can be overriden.

        Parameters
        ----------
        value : Any
            Value to test for applicability

        Returns
        -------
        bool
            If the type us applicable.
        """
        return isinstance(value, tuple(self.applicable_backward_types()))

    @abstractmethod
    def forward(self,
                value: Any,
                name: str,
                object_context: Dict[str, Any],
                handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
                memo: Optional[Dict[Any, UUID]] = None,
                **kwargs
                ) -> Any:
        """Forward path defined the runtime value to a value which can be saved in the json format.
        The output of the forward path is expected to be directly serializable.
        
        Parameters
        ----------
        value : Any
            The actual value wich should be converted.
        name : str
            The name of the property.
        object_context : Dict[str, Any]
            The object context, like other elements which are saved.
        handle_unmatched : Literal[&#39;identity&#39;, &#39;raise&#39;, &#39;jsonpickle&#39;]
            What to do if no rule is matching (nested behavior)
        memo : Dict[Any, UUID]
            Containing already serialized objects. 
            Can be used to avoid cycles when serializing.
            UUID will be used to uniquely identify objects.
        Returns
        -------
        Any
            Some converted object.
        """
        ...

    def backward(self, value: Any, **kwargs) -> Any:
        """The backward path, for objects which must artificially further processed as they
        are maybe wrappers e.g.

        Parameters
        ----------
        value : Any
            The value to convert backward to its original state.

        Returns
        -------
        Any
            Converted object.
        """
        return value

