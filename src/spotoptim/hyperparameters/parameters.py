from typing import List, Dict, Any, Optional, Tuple, Union
from .repr_helpers import Parameter, Bounds


class ParameterSet:
    """
    User-friendly interface for defining hyperparameters.
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
    ):
        """Add a float hyperparameter.

        Args:
            name: Name of the parameter.
            low: Lower bound.
            high: Upper bound.
            default: Default value.
            transform: Transformation string, e.g., "log", "log(x)", "pow(x, 2)".
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

    def add_int(self, name: str, low: int, high: int, default: Optional[int] = None):
        """Add an integer hyperparameter."""
        self._parameters.append(
            {"name": name, "type": "int", "low": low, "high": high, "default": default}
        )
        self._var_names.append(name)
        self._var_types.append("int")
        self._bounds.append((low, high))
        self._var_trans.append(None)
        if default is not None:
            self._defaults[name] = default
        return self

    def add_categorical(
        self, name: str, choices: List[str], default: Optional[str] = None
    ):
        """Add a categorical hyperparameter."""
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
        """Returns bounds formatted for SpotOptim."""
        return self._bounds

    @property
    def var_type(self) -> List[str]:
        """Returns variable types formatted for SpotOptim."""
        return self._var_types

    @property
    def var_name(self) -> List[str]:
        """Returns variable names."""
        return self._var_names

    @property
    def var_trans(self) -> List[Optional[str]]:
        """Returns variable transformations."""
        return self._var_trans

    def sample_default(self) -> Dict[str, Any]:
        """Returns the default configuration."""
        return self._defaults.copy()

    def names(self) -> List[str]:
        return self._var_names

    def __repr__(self) -> str:
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
