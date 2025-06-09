from abc import ABC, abstractmethod


class Operator(ABC):
    """
    Abstract base class for defining common transformation operations.
    Subclasses must implement transform and transposed_transform methods.
    """

    @abstractmethod
    # def transform(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    def transform(self, x: any, *args, **kwargs) -> any:
        """
        Applies a transformation to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor to transform.
            *args, **kwargs: Additional parameters for the transsformation.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        pass

    @abstractmethod
    # def transposed_transform(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    def transposed_transform(self, x: any, *args, **kwargs) -> any:
        """
        Applies the transposed version of the transformation to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor to apply the transposed transform.
            *args, **kwargs: Additional parameters for the transformation.

        Returns:
            torch.Tensor: The transposed transformed tensor.
        """
        pass
