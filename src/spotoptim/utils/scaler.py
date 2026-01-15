import torch


class TorchStandardScaler:
    """
    A class for scaling data using standardization with torch tensors.
    This scaler computes the mean and standard deviation on a dataset so that
    it can later be used to scale the data using the computed mean and standard deviation.

    Attributes:
        mean (torch.Tensor): The mean value computed over the fitted data.
        std (torch.Tensor): The standard deviation computed over the fitted data.

    Examples:
        >>> import torch
        >>> from spotoptim.utils.scaler import TorchStandardScaler
        >>> # Create a sample tensor
        >>> tensor = torch.rand((10, 3))  # Random tensor with shape (10, 3)
        >>> scaler = TorchStandardScaler()
        >>> # Fit the scaler to the data
        >>> scaler.fit(tensor)
        >>> # Transform the data using the fitted scaler
        >>> transformed_tensor = scaler.transform(tensor)
        >>> print(transformed_tensor.shape)
        torch.Size([10, 3])
        >>> # Using fit_transform method to fit and transform in one step
        >>> another_tensor = torch.rand((10, 3))
        >>> scaled_tensor = scaler.fit_transform(another_tensor)
    """

    def __init__(self):
        """
        Initializes the TorchStandardScaler class without any pre-defined mean and std.
        """
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor) -> None:
        """
        Compute the mean and standard deviation of the input tensor.

        Args:
            x (torch.Tensor): The input tensor, expected shape [n_samples, n_features]

        Raises:
            TypeError: If the input is not a torch tensor.
        """
        if not torch.is_tensor(x):
            raise TypeError("Input should be a torch tensor")
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, unbiased=False, keepdim=True)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scale the input tensor using the computed mean and standard deviation.

        Args:
            x (torch.Tensor): The input tensor to be transformed, expected shape [n_samples, n_features]

        Returns:
            torch.Tensor: The scaled tensor.

        Raises:
            TypeError: If the input is not a torch tensor.
            RuntimeError: If the scaler has not been fitted before transforming data.
        """
        if not torch.is_tensor(x):
            raise TypeError("Input should be a torch tensor")
        if self.mean is None or self.std is None:
            raise RuntimeError("Must fit scaler before transforming data")
        x = (x - self.mean) / (self.std + 1e-7)
        return x

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fit the scaler to the input tensor and then scale the tensor.

        Args:
            x (torch.Tensor): The input tensor, expected shape [n_samples, n_features]

        Returns:
            torch.Tensor: The scaled tensor.

        Raises:
            TypeError: If the input is not a torch tensor.
        """
        self.fit(x)
        return self.transform(x)
