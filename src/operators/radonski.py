from skimage.transform import radon, iradon
from src.operators.operator import Operator
import numpy as np

class Radonski(Operator):
    def __init__(self, angles: np.ndarray) -> None:
        self.angles = angles

    def transform(self, x: np.ndarray) -> np.ndarray:
        return radon(image=x,
                     theta=self.angles,
                     circle=False,
                     preserve_range=True)

    def transposed_transform(self, y: np.ndarray) -> np.ndarray:
        # This must be here, at the same indent as transform()
        return iradon(y,
                      theta=self.angles,
                      filter_name=None,
                      circle=False,
                      preserve_range=True)

    def fbp(self, y: np.ndarray) -> np.ndarray:
        return iradon(y,
                      theta=self.angles,
                      filter_name='ramp',
                      circle=False,
                      preserve_range=True)
    


