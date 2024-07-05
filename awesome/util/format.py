import copy
from enum import Enum
import math
import re
from datetime import timedelta
from string import Template
from typing import Any, Dict, Literal, Optional, Tuple, Type, TypeVar, Union
from awesome.util.logging import logger
import inspect
from pandas import Series

from awesome.error import ArgumentNoneError
from awesome.util.reflection import dynamic_import

CAMEL_SEPERATOR_PATTERN = re.compile(r'((?<!^)(?<!_))((?=[A-Z][a-z])|((?<=[a-z])(?=[A-Z])))')
UPPER_SNAKE_PATTERN = re.compile(r'^([A-Z]+_?)*([A-Z]+)$')
UPPER_PATTERN = re.compile(r'^([A-Z]+)$')

REGEX_ISO_8601_PATTERN = r'^(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9]):([0-5][0-9]):([0-5][0-9])(\.[0-9]+)?(Z|[+-](?:2[0-3]|[01][0-9]):[0-5][0-9])?$'
REGEX_ISO_COMPILED = re.compile(REGEX_ISO_8601_PATTERN)


def to_snake_case(input: str) -> str:
    """Converts a upper snake case, or camel case pattern to lower snake case.

    Parameters
    ----------
    input : str
        The input string to convert.

    Returns
    -------
    str
        The converted string

    Raises
    ------
    ArgumentNoneError
        If input is None.
    ValueError
        If input is not a string.
    """
    if input is None:
        raise ArgumentNoneError("input")
    if not isinstance(input, str):
        raise ValueError(f"Type of input should be string but is: {type(input).__name__}")
    if ('_' not in input or not UPPER_SNAKE_PATTERN.match(input)) and not UPPER_PATTERN.match(input):
        return CAMEL_SEPERATOR_PATTERN.sub('_', input).lower()
    else:
        return input.lower()
    

def snake_to_upper_camel(input: str, sep: str = "") -> str:
    """Converts a snake '_' pattern to a upper camel case pattern.

    Parameters
    ----------
    input : str
        The input string to convert.
    sep : str, optional
        Seperator for individual words., by default ""

    Returns
    -------
    str
        The altered string.
    """
    words = [x.capitalize() for x in input.split("_")]
    return sep.join(words)

class TimeDeltaTemplate(Template):
    """Class for formating timedelta with strftime like syntax"""
    delimiter = "%"

def strfdelta(delta: timedelta, format: str) -> str:
    """Formats the timedelta.
    Supported substitudes:

    %D - Days
    %H - Hours
    %M - Minutes
    %S - Seconds

    Parameters
    ----------
    delta : timedelta
        The timedelta to format
    format : str
        The format string.

    Returns
    -------
    str
        The formatted string.
    """
    d = {"D": delta.days}
    d["H"], rem = divmod(delta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    t = TimeDeltaTemplate(format)
    d = {k: f'{v:02d}' for k, v in d.items()}
    return t.substitute(**d)


def destinctive_number_float_format(values: Series,
                                    max_decimals: int = 10,
                                    use_scientific_format: Optional[bool] = None,
                                    distinctive_digits_for_scientific_format: int = 5
                                    ) -> str:
    """Evaluates based on the number of destinctive digits the best format for a float.

    Parameters
    ----------
    values : Series
        The values to evaluate.
    max_decimals : int, optional
        The upper limit for decimal digits, by default 10
    use_scientific_format : Optional[bool], optional
        If the scientific format should be used or not, by default decision will be based 
        on how many distinctive decimal places are needed., by default None
    distinctive_digits_for_scientific_format : int, optional
        The number of destinctive digits for the scientific format, by default 4
        If a number needs more or equal destinctive digits to judge wether there are different
         it will be formatted in scientific format.
    Returns
    -------
    str
        The string format.
    """
    destinctive = False
    values = copy.deepcopy(values)

    # Ignore Nan

    values = values.dropna()

    def _count_leading_zeros(value: float):
        if value >= 1:
            return 0
        post = str(value).split('.')[1]
        i = 0
        while post[i] == '0':
            i += 1
        return i

    leading_zeros = min(values.apply(lambda x: _count_leading_zeros(x)))
    max_leading_zeros = max(values.apply(lambda x: _count_leading_zeros(x)))

    unshifted = copy.deepcopy(values)
    unshifted_i = 0

    _exp = dict()

    # Move large numbers behind the comma
    for i, _v in values.items():
        exp = 0
        num = _v
        while num >= 10:
            num = num / 10
            exp += 1
        while num < 1:
            num = num * 10
            exp -= 1
        values[i] = num
        _exp[i] = exp

    i = 0
    float_destinctive = False
    while (not destinctive or not float_destinctive) and i < max_decimals:
        if not destinctive:
            _v = values * 10**i
            _v = _v.apply(math.floor)
            _cmp = Series({k: val * (10 ** _exp[k]) for k, val in _v.items()})
            if len(_cmp.unique()) == len(values.unique()):
                destinctive = True
            else:
                i += 1
        if len(set([round(x, unshifted_i) for x in unshifted])) != len(unshifted.unique()):
            unshifted_i += 1
        else:
            float_destinctive = True

    _exp_max = max(list(_exp.values()))
    _exp_min = min(list(_exp.values()))
    _exp_mean = sum(list(_exp.values())) / len(_exp.values())

    if ((_exp_max - _exp_min) >= 3 or _exp_mean < -2) and use_scientific_format is None:
        use_scientific_format = True
    else:
        use_scientific_format = False
    num_destinctive_digits = i if use_scientific_format else unshifted_i
    return f"{{:.{num_destinctive_digits}{'e' if use_scientific_format else 'f'}}}"


def latex_postprocessor(text: str, 
            replace_underline: bool = True,
            replace_bfseries: bool = True,
            replace_text_decoration_underline: bool = True,
            replace_booktabs_rules: bool = True
            ) -> str:
    """Postprocesses a latex string.
    Can applied to pandas to latex commands to fix incorrect latex syntax.

    Parameters
    ----------
    text : str
        The string which shoud be altered.
    replace_underline : bool, optional
        If underlines should be replaced. Will replace all and does not care for math mode., by default True
    replace_bfseries : bool, optional
        Replace a bfseries which should be a textbf, by default True

    Returns
    -------
    str
        The processed string.
    """    
    # Pattern
    UNDERSCORE_IN_TEXT = r"(?<=([A-z0-9\_]))\_(?=[A-z0-9\_])"
    BF_SERIES = r"(\\bfseries)( )(?P<text>[A-z0-9.\-\_\+]+)( )"
    TEXT_DECO_UNDERLINE = r"(\\text-decorationunderline)( )(?P<text>[A-z0-9.\-\_\+]+)( )"

    if replace_underline:
        text = re.sub(UNDERSCORE_IN_TEXT, r"\_", text)
    if replace_bfseries:
        text = re.sub(BF_SERIES, r"\\textbf{\g<text>}", text)
    if replace_text_decoration_underline:
        text = re.sub(TEXT_DECO_UNDERLINE, r"\\underline{\g<text>}", text)
    if replace_booktabs_rules:
        text = text.replace("\\toprule", "\\hline")
        text = text.replace("\\midrule", "\\hline")
        text = text.replace("\\bottomrule", "\\hline")
    return text

E = TypeVar('E', bound=Enum)

def parse_enum(cls: Type[E], value: Any) -> E:
    """Parses a value to an enum.
    Simple helper function to parse a value to an enum.

    Parameters
    ----------
    cls : Type[E]
        Type of the enum.
    value : Any
        Value to parse.

    Returns
    -------
    E
        The parsed enum value.

    Raises
    ------
    ValueError
        If the value is not of the correct type or cannot be parsed.
    """
    if not issubclass(cls, Enum):
        raise ValueError(f"Type of cls should be an Enum but is: {type(cls).__name__}")
    if isinstance(value, cls):
        return value
    elif isinstance(value, (str, int)):
        return cls(value)
    else:
        raise ValueError(f"Type of value for creating: {cls.__name__} should be either string or int but is: {type(value).__name__}")
        

def parse_type(_type_or_str: Union[Type, str], 
               parent_type: Optional[Union[Type, Tuple[Type, ...]]] = None,
               instance_type: Optional[Type] = None,
               variable_name: Optional[str] = None,
               default_value: Optional[Any] = None,
               handle_invalid: Literal["set_default", "raise", "set_none"] = "raise" ,
               handle_not_a_class: Literal["ignore", "raise"] = "raise"
               ) -> Type:
    """Parses a type from a string or type.
    Optioanally includes checks for beeing a subclass of a parent type or an instance of a type.

    Parameters
    ----------
    _type_or_str : Union[Type, str]
        The type or string to parse.
    parent_type : Optional[Union[Type, Tuple[Type, ...]]], optional
        If the type is subclass of any of the given types, by default None
    instance_type : Optional[Type], optional
        If the type is an instance of some type, by default None
    variable_name : Optional[str], optional
        The name of the variable. Can be used to further specify the error message, by default None
    default_value : Optional[Any], optional
        The default value which should be used when parsing fails and the handle invalid mode is set default, by default None
    handle_invalid : Literal[&quot;set_default&quot;, &quot;raise&quot;, &quot;set_none&quot;], optional
        How an invalid type or error during parsing should be handled, by default "raise"
        "raise" - Raises an error
        "set_default" - Sets the default value
        "set_none" - Sets None as return value
        Both non raise options will log a warning.
    handle_not_a_class : Literal[&quot;ignore&quot;, &quot;raise&quot;], optional
        How to handle if the type is not a class, by default "raise"
        "ignore" - Will ignore the error and check for isinstance
        "raise" - Will raise an error
        
    Returns
    -------
    Type
        The parsed and checked type.
    """
    def handle_error(error_message: str, exception: Optional[Exception] = None) -> Any:
        prefix = f"Failing to parse type of variable {variable_name}." if variable_name is not None else "Failing to parse type."
        if handle_invalid == "raise":
            if exception is not None:
                raise ValueError(error_message) from exception
            else:
                raise ValueError(error_message)
        elif handle_invalid == "set_default":
            logger.warning(prefix + error_message + f" Setting default value: {default_value}")
            return default_value
        elif handle_invalid == "set_none":
            logger.warning(prefix + error_message + " Setting None")
            return None
        else:
            raise ValueError(prefix + f"Invalid handle_invalid: {handle_invalid}")
    _parsed_type = None
    if isinstance(_type_or_str, str):
        try:
            _parsed_type = dynamic_import(_type_or_str)
        except ImportError as e:
            return handle_error(f"Could not import: {_type_or_str} due to an {e.__class__.__name__} does the Module / Type exists and is it installed?", e)
    elif isinstance(_type_or_str, type):
        _parsed_type = _type_or_str
    else:
        return handle_error(f"Type of _type_or_str should be either string or type but is: {type(_type_or_str).__name__}")
    check_instance = True
    if parent_type is not None:
        if inspect.isclass(_parsed_type):
            if not issubclass(_parsed_type, parent_type):
                return handle_error(f"Type: {_parsed_type.__name__} is not a subclass of (any): {parent_type}")
            check_instance = False
        else:
            if handle_not_a_class == "raise":
                return handle_error(f"Type: {_parsed_type.__name__} is not a class.")
            elif handle_not_a_class == "ignore":
                check_instance = True
            else:
                raise ValueError(f"Invalid handle_not_a_class: {handle_not_a_class}")
            
    if instance_type is not None and check_instance:
        if not isinstance(_parsed_type, instance_type):
            return handle_error(f"Type: {_parsed_type.__name__} is not an instance of: {instance_type.__name__}")
    return _parsed_type