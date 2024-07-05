from typing import Any, Dict, List, Literal, Optional, Type

from awesome.error.argument_none_error import ArgumentNoneError
from .json_serialization_rule import JsonSerializationRule
import sys


class JsonSerializationRuleRegistry():
    """Defines the known rules for converting an object into a json convertible structure."""

    __instance__ = None
    """Singleton instance."""

    __simple_type_rules_forward__: Dict[Type, JsonSerializationRule]
    __simple_type_rules_backward__: Dict[Type, JsonSerializationRule]

    __simple_non_type_rules__: List[JsonSerializationRule]

    __complex_type_rules_forward__: Dict[Type, JsonSerializationRule]
    __complex_type_rules_backward__: Dict[Type, JsonSerializationRule]

    __complex_non_type_rules__: List[JsonSerializationRule]

    __identity_rule__: JsonSerializationRule
    __json_pickle_rule__: JsonSerializationRule

    @classmethod
    def instance(cls) -> 'JsonSerializationRuleRegistry':
        if cls.__instance__ is None:
            cls.__instance__ = JsonSerializationRuleRegistry()
        return cls.__instance__

    def __init__(self) -> None:
        self.__identity_rule__ = None
        self.__json_pickle_rule__ = None
        self.__simple_rules__ = self._find_simple_rules()
        self.__complex_rules__ = self._find_complex_rules()
        self.__simple_type_rules_forward__, self.__simple_type_rules_backward__ = self._setup_simple_rules()
        self.__complex_type_rules_forward__, self.__complex_type_rules_backward__ = self._setup_complex_rules()
        from awesome.serialization.rules import (
            JsonDictSerializationRule,
            JsonPickleSerializationRule
        )
        self.__complex_non_type_rules__ = [JsonDictSerializationRule()]
        self.__simple_non_type_rules__ = []

    def _find_simple_rules(self, additional_simple_rules: Optional[List[Type[JsonSerializationRule]]] = None) -> List[Type[JsonSerializationRule]]:
        from awesome.serialization.rules import (
            JsonIdentitySerializationRule,
            JsonComplexSerializationRule,
            JsonDatetimeSerializationRule,
            JsonDecimalSerializationRule,
            JsonEnumSerializationRule,
            JsonFunctionSerializationRule,
            JsonConvertibleSerializationRule,
            JsonSequentialSerializationRule,
            JsonModuleListSerializationRule,
            JsonTensorSerializationRule,
            JsonTorchDtypeSerializationRule,
            JsonTorchDeviceSerializationRule,
            JsonTypeSerializationRule,
            JsonSliceSerializationRule
        )
        all_simple_rules: List[Type[JsonSerializationRule]] = [
            JsonIdentitySerializationRule,
            JsonComplexSerializationRule,
            JsonDatetimeSerializationRule,
            JsonDecimalSerializationRule,
            JsonEnumSerializationRule,
            JsonFunctionSerializationRule,
            JsonConvertibleSerializationRule,
            JsonSequentialSerializationRule,
            JsonModuleListSerializationRule,
            JsonTensorSerializationRule,
            JsonTorchDtypeSerializationRule,
            JsonTorchDeviceSerializationRule,
            JsonTypeSerializationRule,
            JsonSliceSerializationRule
        ]

        if 'numpy' in sys.modules:
            from awesome.serialization.rules.numpy import (
                JsonGenericSerializationRule,
                JsonNDArraySerializationRule,
            )
            all_simple_rules = [JsonGenericSerializationRule, JsonNDArraySerializationRule] + all_simple_rules

        if additional_simple_rules is not None:
            all_simple_rules += additional_simple_rules

        # purge duplicates
        all_simple_rules = list(set(all_simple_rules))

        all_simple_rules = sorted(all_simple_rules, key=lambda v: v().priority)
        return all_simple_rules

    def register_simple_rules(self, rules: List[Type[JsonSerializationRule]]):
        """Method to registering simple serialization rules for other packages.

        Parameters
        ----------
        rules : List[Type[JsonSerializationRule]]
            The rules to register

        Raises
        ------
        ArgumentNoneError
            If rules is None.
        ValueError
            If rules are not inheriting von JsonSerializationRule
        """
        if rules is None:
            raise ArgumentNoneError("rules")
        if len(rules) == 0:
            return
        if any((not issubclass(x, JsonSerializationRule) for x in rules)):
            raise ValueError("Some rules not subclassing JsonSerializationRule, cannot register them!")
        # Adding the new rules and keeping others, duplicates are purged.
        self.__simple_rules__ = self._find_simple_rules(rules + self.__simple_rules__)
        self.__simple_type_rules_forward__, self.__simple_type_rules_backward__ = self._setup_simple_rules()

    def register_complex_rules(self, rules: List[Type[JsonSerializationRule]]):
        """Method to registering complex serialization rules for other packages.

        Parameters
        ----------
        rules : List[Type[JsonSerializationRule]]
            The rules to register

        Raises
        ------
        ArgumentNoneError
            If rules is None.
        ValueError
            If rules are not inheriting von JsonSerializationRule
        """
        if rules is None:
            raise ArgumentNoneError("rules")
        if len(rules) == 0:
            return
        if any((not issubclass(x, JsonSerializationRule) for x in rules)):
            raise ValueError("Some rules not subclassing JsonSerializationRule, cannot register them!")
        # Adding the new rules and keeping others, duplicates are purged.
        self.__complex_rules__ = self._find_simple_rules(rules + self.__complex_rules__)
        self.__complex_type_rules_forward__, self.__complex_type_rules_backward__ = self._setup_complex_rules()

    def _find_complex_rules(self, additional_complex_rules: Optional[List[Type]] = None) -> List[Type]:
        from awesome.serialization.rules import (
            JsonListSerializationRule,
            JsonTupleSerializationRule,
            JsonSetSerializationRule,
            JsonDictSerializationRule,
            JsonPickleSerializationRule,
        )

        all_complex_rules: List[Type[JsonSerializationRule]] = [
            JsonTupleSerializationRule,
            JsonSetSerializationRule,
            JsonListSerializationRule,
            JsonDictSerializationRule,
            JsonPickleSerializationRule
        ]

        if additional_complex_rules is not None:
            all_complex_rules += additional_complex_rules

        # purge duplicates
        all_complex_rules = list(set(all_complex_rules))

        all_complex_rules = sorted(all_complex_rules, key=lambda v: v().priority)

        return all_complex_rules

    def _setup_simple_rules(self) -> Dict[Type, JsonSerializationRule]:
        from awesome.serialization.rules import (
            JsonIdentitySerializationRule
        )
        all_simple_rules = self.__simple_rules__

        ret_fwd = {}
        ret_bk = {}

        for rule_type in all_simple_rules:
            rule = rule_type()
            if isinstance(rule, JsonIdentitySerializationRule):
                self.__identity_rule__ = rule
            for apt in rule_type.applicable_forward_types():
                if apt not in ret_fwd:
                    ret_fwd[apt] = rule

        for rule_type in all_simple_rules:
            rule = rule_type()
            for apt in rule_type.applicable_backward_types():
                if apt not in ret_bk:
                    ret_bk[apt] = rule
        return ret_fwd, ret_bk

    def get_simple_rule_forward(self, value: Any) -> JsonSerializationRule:
        """Returns a simple conversion rule if registered for a given value.

        Parameters
        ----------
        value : Any
            Value which should be serialized.

        Returns
        -------
        JsonSerializationRule
            The rule which can perform serialization
        """
        if value is None:
            return None
        if type(value) in self.__simple_type_rules_forward__:
            return self.__simple_type_rules_forward__[type(value)]
        for _type, rule in self.__simple_type_rules_forward__.items():
            if isinstance(value, _type):
                return rule
        for rule in self.__simple_non_type_rules__:
            if rule.is_forward_applicable(value):
                return rule
        return None  # No rule found

    def get_rule_forward(self, value: Any) -> Optional[JsonSerializationRule]:
        """Returns a conversion rule if registered for a given value.

        Parameters
        ----------
        value : Any
            Value which should be serialized.

        Returns
        -------
        Optional[JsonSerializationRule]
            The conversion rule or none of not found.
        """
        rule = JsonSerializationRuleRegistry.instance().get_simple_rule_forward(value)
        if rule is not None:
            return rule
        rule = JsonSerializationRuleRegistry.instance().get_complex_rule_forward(value)
        return rule

    def get_simple_rule_backward(self, value: Any) -> JsonSerializationRule:
        if value is None:
            return None
        if type(value) in self.__simple_type_rules_backward__:
            return self.__simple_type_rules_backward__[type(value)]
        for _type, rule in self.__simple_type_rules_backward__.items():
            if isinstance(value, _type):
                return rule
        for rule in self.__simple_non_type_rules__:
            if rule.is_backward_applicable(value):
                return rule
        return None  # No rule found

    def get_rule_backward(self, value: Any) -> Optional[JsonSerializationRule]:
        """Returns a conversion rule if registered for a given value.

        Parameters
        ----------
        value : Any
            Value which should be deserialized.

        Returns
        -------
        Optional[JsonSerializationRule]
            The conversion rule or none of not found.
        """
        rule = JsonSerializationRuleRegistry.instance().get_simple_rule_backward(value)
        if rule is not None:
            return rule
        rule = JsonSerializationRuleRegistry.instance().get_complex_rule_backward(value)
        return rule

    def get_complex_rule_forward(self, value: Any) -> JsonSerializationRule:
        """Returns a complex conversion rule if registered for a given value.

        Parameters
        ----------
        value : Any
            Value which should be serialized.

        Returns
        -------
        JsonSerializationRule
            The rule which can perform serialization
        """
        if value is None:
            return None
        if type(value) in self.__complex_type_rules_forward__:
            return self.__complex_type_rules_forward__[type(value)]
        for _type, rule in self.__complex_type_rules_forward__.items():
            if isinstance(value, _type):
                return rule
        for rule in self.__complex_non_type_rules__:
            if rule.is_forward_applicable(value):
                return rule
        return None  # No rule found

    def get_default_rule_forward(self, unmatched_rule: Literal['identity', 'jsonpickle']):
        if unmatched_rule == 'identity':
            return self.__identity_rule__
        elif unmatched_rule == 'jsonpickle':
            return self.__json_pickle_rule__
        else:
            raise ValueError("Unknown behavior.")

    def get_complex_rule_backward(self, value: Any) -> JsonSerializationRule:
        if value is None:
            return None
        if type(value) in self.__complex_type_rules_backward__:
            return self.__complex_type_rules_backward__[type(value)]
        for _type, rule in self.__complex_type_rules_backward__.items():
            if isinstance(value, _type):
                return rule
        for rule in self.__complex_non_type_rules__:
            if rule.is_backward_applicable(value):
                return rule
        return None  # No rule found

    def _setup_complex_rules(self) -> Dict[Type, JsonSerializationRule]:
        from awesome.serialization.rules import (
            JsonPickleSerializationRule
        )

        all_complex_rules = self.__complex_rules__

        ret_fwd = {}
        ret_bk = {}

        for rule_type in all_complex_rules:
            rule = rule_type()
            if isinstance(rule, JsonPickleSerializationRule):
                self.__json_pickle_rule__ = rule
            for apt in rule_type.applicable_forward_types():
                if apt not in ret_fwd:
                    ret_fwd[apt] = rule

        for rule_type in all_complex_rules:
            rule = rule_type()
            for apt in rule_type.applicable_backward_types():
                if apt not in ret_bk:
                    ret_bk[apt] = rule
        return ret_fwd, ret_bk
