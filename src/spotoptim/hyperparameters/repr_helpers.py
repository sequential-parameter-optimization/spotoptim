from typing import Any, Union, List
from dataclasses import dataclass


@dataclass
class Bounds:
    low: Any
    high: Any

    def __repr__(self):
        return f"Bounds(low={self.low!r}, high={self.high!r})"


@dataclass
class Parameter:
    name: str
    var_name: str
    bounds: Union[Bounds, List[str], tuple]
    default: Any
    log: bool
    type: str

    def __repr__(self):
        return (
            f"Parameter(\n"
            f"        name={self.name!r},\n"
            f"        var_name={self.var_name!r},\n"
            f"        bounds={self.bounds!r},\n"
            f"        default={self.default!r},\n"
            f"        log={self.log!r},\n"
            f"        type={self.type!r}\n"
            f"    )"
        )
