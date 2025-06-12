from abc import ABC, abstractmethod
# import torch

class Sparsifier(ABC):
    """
    Abstract base class for defining common transformation operations.
    Subclasses must implement transform and transposed_transform methods.
    """

    @abstractmethod
    def transform(self, x: any, *args, **kwargs) :
        """
        Applies a transformation to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor to transform.
            *args, **kwargs: Additional parameters for the transformation.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        pass

    @abstractmethod
    def transposed_transform(self, x: any, *args, **kwargs) :
        """
        Applies the transposed version of the transformation to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor to apply the transposed transform.
            *args, **kwargs: Additional parameters for the transformation.

        Returns:
            torch.Tensor: The transposed transformed tensor.
        """
        pass

    @abstractmethod
    def transform_abs(self, x: any, *args, **kwargs) :
        """
        Applies a transformation (with absolute values) to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor to transform.
            *args, **kwargs: Additional parameters for the transformation.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        pass

    @abstractmethod
    def transposed_transform_abs(self, x: any, *args, **kwargs) -> torch.Tensor:
        """
        Applies the transposed version of the transformation (with absolute values) to the input tensor.
        
        Args:
            x (torch.Tensor): The input tensor to apply the transposed transform.
            *args, **kwargs: Additional parameters for the transformation.

        Returns:
            torch.Tensor: The transposed transformed tensor.
        """
        pass