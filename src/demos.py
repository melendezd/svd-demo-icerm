from __future__ import annotations

from enum import Enum

from nptyping import NDArray, UInt8  # type: ignore
from typing import Any, Callable, TypeVar, Generic, Type, Dict

import PIL.Image  # type: ignore
import numpy as np  # type: ignore

import streamlit as st  # type: ignore

import matrix_like as matl

import constants


class DemoName(Enum):
    """Identifiers for SVD demo pages"""
    image_compression = "Image Compression"
    audio_compression = "Audio Compression"
    video_bg          = "Video Background Extraction"
    watermarking      = "Digital Watermarking"


class Demo(matl.IDisplayable):
    pass


M = TypeVar('M', bound=matl.MatrixLikeDisplayableLoadable)
class SVDDemo(Generic[M], Demo):
    '''
    A demo showcasing compression of a matrix-like object
    using low-rank approximations.
    '''
    def __init__(self, demo_type: Type[M], examples: Dict[str, str]):
        self.demo_type = demo_type
        self.example_dict: Dict[str, M] = \
            {label: demo_type.load(constants.RESOURCE_PATH + fname)
             for label, fname in examples.items()}

    def display(self) -> None:
        selection = st.sidebar.selectbox(
            label='Select an example',
            options=list(self.example_dict.keys()))
        self.example_dict[selection].display()

# Dictionary mapping image names to file paths
image_dict = {'Raccoon': 'images/raccoon.jpg', 'Fox': 'images/fox.jpg'}

image_compression_demo = SVDDemo[matl.Image](matl.Image, image_dict)

# Dictionary mapping demo selections to demos
demo_dict: Dict[DemoName, Demo] = {DemoName.image_compression: image_compression_demo}
