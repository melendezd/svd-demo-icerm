from __future__ import annotations

from nptyping import NDArray, UInt8
from typing import Any, Callable

import PIL.Image
import numpy as np

import streamlit as st

class IMatrixLike:
    def to_matrix(self) -> NDArray:
        raise NotImplementedError

    def compute(self, fun: Callable[NDArray, NDArray]) -> IMatrixLike:
        return from_matrix(fun(to_matrix(self)))

    @staticmethod
    def from_matrix(obj: NDArray[(Any,  Any)], **kwargs) -> IMatrixLike:
        raise NotImplementedError


class IDisplayable:
    def display(self) -> None:
        raise NotImplementedError


class Image(IMatrixLike, IDisplayable):
    def __init__(self, ndim: UInt8, img: NDArray[(Any, Any, ndim), UInt8]):
        self.img = img
        self.ndim = ndim

    def to_matrix(self) -> NDArray[(Any, Any), UInt8]:
        '''
        Flattens an image with any number of channels into a 2D matrix
        '''
        return self.img.reshape(self.img.shape[0], self.img.shape[1] * self.ndim)

    @staticmethod
    def from_matrix(img_matrix: NDArray[(Any, Any), UInt8], **kwargs) -> Image:
        '''
        Converts a 2D array to an image given the number of channels as ndim in kwargs
        '''
        ndim: UInt8 = kwargs.get('ndim')
        return Image(ndim,
                     img_matrix.reshape(img_matrix.shape[0],
                                        img_matrix.shape[1]//ndim, ndim))

    def compute(self, fun: Callable[NDArray, NDArray]) -> Image:
        '''
        Applies a function to the flattened version of an image, rounds clips the
        resulting values between 0 and 255, and returns the resulting image
        '''
        fun_img_mat = fun(self.to_matrix())
        fun_img_mat_normalized = np.clip(fun_img_mat, 0, 255).astype(np.uint8)
        return Image.from_matrix(fun_img_mat_normalized, ndim=self.ndim)

    def display(self):
        '''
        Displays the image using Streamlit
        '''
        st.image(self.img)

    @staticmethod
    def load(path: str) -> Image:
        img_arr = np.asarray(PIL.Image.open(path))
        return Image(img_arr.ndim, np.asarray(PIL.Image.open(path)))

