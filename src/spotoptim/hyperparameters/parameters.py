from typing import List, Dict, Any, Optional, Tuple, Union
from .repr_helpers import Parameter, Bounds


class ParameterSet:
    """
    User-friendly interface for defining hyperparameters.

    This class allows for the definition of a set of hyperparameters including their types,
    bounds, default values, and transformations. It supports float, integer, and categorical
    parameters and provides a fluent interface for chaining parameter definitions.

    Attributes:
        _parameters (List[Dict]): List of parameter definitions.
        _var_names (List[str]): List of parameter names.
        _var_types (List[str]): List of parameter types ('float', 'int', 'factor').
        _bounds (List[Union[Tuple, List]]): List of bounds for each parameter.
        _defaults (Dict[str, Any]): Dictionary of default values.
        _var_trans (List[Optional[str]]): List of variable transformations.

    Examples:
        >>> ps = ParameterSet()
        >>> ps.add_float("max_depth", 1, 10, default=3)
        ParameterSet(
            max_depth=Parameter(name='max_depth', type='float', check_on_set=True, bounds=(1, 10), default=3),
        )
    """

    def __init__(self):
        self._parameters = []
        self._var_names = []
        self._var_types = []
        self._bounds = []
        self._defaults = {}
        self._var_trans = []

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        default: Optional[float] = None,
        transform: Optional[str] = None,
    ) -> "ParameterSet":
        """Add a float hyperparameter.

        Args:
            name: Name of the parameter.
            low: Lower bound of the parameter range.
            high: Upper bound of the parameter range.
            default: Default value for the parameter.
            transform: Transformation string, e.g., "log", "log(x)", "pow(x, 2)".

        Returns:
            ParameterSet: The instance itself to allow method chaining.

        Examples:
            >>> from spotoptim.hyperparameters import ParameterSet
            >>> ps = ParameterSet()
            >>> ps.add_float("learning_rate", 0.0001, 0.1, default=0.01, transform="log")
        """
        self._parameters.append(
            {
                "name": name,
                "type": "float",
                "low": low,
                "high": high,
                "default": default,
                "transform": transform,
            }
        )
        self._var_names.append(name)
        self._var_types.append("float")
        self._bounds.append((low, high))
        self._var_trans.append(transform)
        if default is not None:
            self._defaults[name] = default
        return self

    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        default: Optional[int] = None,
        transform: Optional[str] = None,
    ) -> "ParameterSet":
        """Add an integer hyperparameter.

        Args:
            name: Name of the parameter.
            low: Lower bound of the parameter range (inclusive).
            high: Upper bound of the parameter range (inclusive).
            default: Default value for the parameter.
            transform: Transformation string, e.g., "log", "log(x)", "pow(x, 2)".

        Returns:
            ParameterSet: The instance itself to allow method chaining.

        Examples:
            >>> from spotoptim.hyperparameters import ParameterSet
            >>> ps = ParameterSet()
            >>> ps.add_int("n_estimators", 10, 100, default=50)
            >>> ps.add_int("epochs", 2, 5, transform="log")
        """
        self._parameters.append(
            {
                "name": name,
                "type": "int",
                "low": low,
                "high": high,
                "default": default,
                "transform": transform,
            }
        )
        self._var_names.append(name)
        self._var_types.append("int")
        self._bounds.append((low, high))
        self._var_trans.append(transform)
        if default is not None:
            self._defaults[name] = default
        return self

    def add_factor(
        self, name: str, choices: List[str], default: Optional[str] = None
    ) -> "ParameterSet":
        """Add a factor (categorical) hyperparameter.

        Args:
            name: Name of the parameter.
            choices: List of possible values for the parameter.
            default: Default value for the parameter.

        Returns:
            ParameterSet: The instance itself to allow method chaining.


        Examples:
            >>> from spotoptim.hyperparameters import ParameterSet
            >>> ps = ParameterSet()
            >>> ps.add_factor("optimizer", ["adam", "sgd"], default="adam")
        """
        self._parameters.append(
            {"name": name, "type": "factor", "choices": choices, "default": default}
        )
        self._var_names.append(name)
        self._var_types.append("factor")
        # For SpotOptim, categorical bounds are the list of choices (for factor detection)
        # SpotOptim checks if bound is tuple/list of strings.
        self._bounds.append(choices)
        self._var_trans.append(None)
        if default is not None:
            self._defaults[name] = default
        return self

    @property
    def bounds(
        self,
    ) -> List[Union[Tuple[float, float], Tuple[int, int], Tuple[str, ...]]]:
        """Returns bounds formatted for SpotOptim.

        Returns:
            List[Union[Tuple[float, float], Tuple[int, int], Tuple[str, ...]]]:
            A list of bounds where each element corresponds to a parameter.
        """
        return self._bounds

    @property
    def var_type(self) -> List[str]:
        """Returns variable types formatted for SpotOptim.

        Returns:
            List[str]: A list of strings representing the types of parameters
            ('float', 'int', 'factor').
        """
        return self._var_types

    @property
    def var_name(self) -> List[str]:
        """Returns variable names.

        Returns:
            List[str]: A list of parameter names.
        """
        return self._var_names

    @property
    def var_trans(self) -> List[Optional[str]]:
        """Returns variable transformations.

        Returns:
            List[Optional[str]]: A list of transformation strings or None for each parameter.
        """
        return self._var_trans

    def sample_default(self) -> Dict[str, Any]:
        """Returns the default configuration.

        Returns:
            Dict[str, Any]: A dictionary mapping parameter names to their default values.

        Examples:
            >>> ps = ParameterSet()
            >>> ps.add_int("x", 1, 10, default=5)
            >>> ps.sample_default()
            {'x': 5}
        """
        return self._defaults.copy()

    def names(self) -> List[str]:
        """Returns a list of parameter names.

        Returns:
            List[str]: The names of the parameters.
        """
        return self._var_names

    def __repr__(self) -> str:
        """String representation of the ParameterSet.

        Returns:
            str: A formatted string showing the parameters in the set.
        """
        lines = ["ParameterSet("]
        for p in self._parameters:
            name = p["name"]

            # Construct Bounds based on type
            if p["type"] == "factor":
                # For factor, bounds are simply the list of choices
                choices = p["choices"]
                bounds = choices
            else:
                b = Bounds(low=p["low"], high=p["high"])
                bounds = b

            param_obj = Parameter(
                name=name,
                var_name=name,
                bounds=bounds,
                default=p.get("default"),
                transform=p.get("transform"),
                type=p["type"],
            )

            # Indent parameter repr
            p_str = repr(param_obj).replace("\n", "\n    ")
            lines.append(f"    {name}={p_str},")

        lines.append(")")
        return "\n".join(lines)
