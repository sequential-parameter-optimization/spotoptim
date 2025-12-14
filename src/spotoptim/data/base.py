import abc
import inspect
import pathlib
import re
from typing import Optional

__all__ = ["Config", "FileConfig"]


class Config(abc.ABC):
    """Base class for all configurations.

    All configurations inherit from this class, be they stored in a file or generated on the fly.

    Attributes:
        desc (str): The description from the docstring.
        _repr_content (dict): The items that are displayed in the __repr__ method.
    """

    def __init__(self):
        """Initialize a Config object."""
        pass

    @property
    def desc(self) -> str:
        """Return the description from the docstring.

        Returns:
            str: The description from the docstring.
        """
        desc = re.split(pattern=r"\w+\n\s{4}\-{3,}", string=self.__doc__, maxsplit=0)[0]
        return inspect.cleandoc(desc)

    @property
    def _repr_content(self) -> dict:
        """The items that are displayed in the __repr__ method.

        This property can be overridden in order to modify the output of the __repr__ method.

        Returns:
            dict: A dictionary containing the items to be displayed in the __repr__ method.
        """
        content = {}
        content["Name"] = self.__class__.__name__
        return content


class FileConfig(Config):
    """Base class for configurations that are stored in a local file.

    Args:
        filename (str): The file's name.
        directory (Optional[str]):
            The directory where the file is contained.
            Defaults to the location of the `datasets` module.
        desc (dict): Extra config parameters to pass as keyword arguments.
    """

    def __init__(self, filename: str, directory: Optional[str] = None, **desc):
        super().__init__(**desc)
        self.filename = filename
        self.directory = directory

    @property
    def path(self) -> pathlib.Path:
        """The path to the configuration file.

        Returns:
            pathlib.Path: The path to the configuration file.
        """
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    @property
    def _repr_content(self) -> dict:
        """The content of the string representation of the FileConfig object.

        Returns:
            dict: A dictionary containing the content of the string representation of the FileConfig object.
        """
        content = super()._repr_content
        content["Path"] = str(self.path)
        return content
