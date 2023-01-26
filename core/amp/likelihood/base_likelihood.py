from abc import ABC, abstractclassmethod
from stat import FILE_ATTRIBUTE_ARCHIVE
from typing import List


class BaseLikelihood(ABC):
    @abstractclassmethod
    def fout(self, w : List[float], y : List, V : List[float]) -> List[float]:
        pass

    @abstractclassmethod
    def dwfout(self, w : List[float], y : List, V : List[float]) -> List[float]:
        pass

    @abstractclassmethod
    def channel(self, w : List[float], y : List, V : List[float]):
        pass
    