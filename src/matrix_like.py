from __future__ import annotations

from nptyping import NDArray, UInt8  # type: ignore
from typing import Any, Callable, TypeVar, Generic

import PIL.Image  # type: ignore
import numpy as np  # type: ignore

import streamlit as st  # type: ignore

T = TypeVar("T")
#S = TypeVar("S", bound=IMatrixLike)
S = TypeVar("S")
#U = TypeVar("S")
class IMatrixLike(Generic[T, S]):
    def to_matrix(self) -> NDArray[(Any, Any), T]:
        raise NotImplementedError

    def compute(self, fun: Callable[[NDArray], NDArray]) -> S:
        return self.from_matrix(fun(self.to_matrix()))

    @staticmethod
    def from_matrix(obj: NDArray[(Any,  Any), T], **kwargs: Any) -> S:
        raise NotImplementedError


class IDisplayable:
    def display(self) -> None:
        raise NotImplementedError

class ILoadable(Generic[T]):
    @staticmethod
    def load(path: str) -> T:
        raise NotImplementedError


class MatrixLikeDisplayableLoadable(Generic[T, S], IMatrixLike[T, S], ILoadable[S],
                                    IDisplayable):
    pass


class Image(MatrixLikeDisplayableLoadable[UInt8, 'Image']):
    def __init__(self, ndim: UInt8, img: NDArray[(Any, Any, Any), UInt8]):
        self.img = img
        self.ndim = ndim

    def to_matrix(self) -> NDArray[(Any, Any), UInt8]:
        '''
        Flattens an image with any number of channels into a 2D matrix
        '''
        return self.img.reshape(self.img.shape[0], self.img.shape[1] * self.ndim)

    def compute(self, fun: Callable[[NDArray], NDArray]) -> Image:
        '''
        Applies a function to the flattened version of an image, rounds clips the
        resulting values between 0 and 255, and returns the resulting image
        '''
        fun_img_mat = fun(self.to_matrix())
        fun_img_mat_normalized = np.clip(fun_img_mat, 0, 255).astype(np.uint8)
        return Image.from_matrix(fun_img_mat_normalized, ndim=self.ndim)

    @staticmethod
    def from_matrix(obj: NDArray[(Any, Any), UInt8], **kwargs: Any) -> Image:
        '''
        Converts a 2D array to an image given the number of channels as ndim in kwargs
        '''
        ndim: UInt8 = kwargs.get('ndim')
        return Image(ndim,
                     obj.reshape(obj.shape[0],
                                        obj.shape[1]//ndim, ndim))


    def display(self):
        '''
        Displays the image using Streamlit
        '''
        st.image(self.img)

    @staticmethod
    def load(path: str) -> Image:
        img_arr = np.asarray(PIL.Image.open(path))
        return Image(img_arr.ndim, np.asarray(PIL.Image.open(path)))

