"""
A web app to demonstrate various applications of the Singular Value
Decomposition (SVD), including:
- Image compression
- Audio compression
- Video background extraction
- Digital watermarking
"""

from enum import Enum

import streamlit as st  # type: ignore
import numpy as np  # type: ignore

from PIL import Image  # type: ignore

import matrix_like as matl

from demos import DemoName
import demos

def main() -> None:
    """ Launch app """
    st.title("Applications of the Singular Value Decomposition")

    demo_selection: DemoName = DemoName(
            st.sidebar.selectbox(
                "Choose a demo",
                [demo_name.value for demo_name in DemoName]
            )
        )

    current_demo = demos.demo_dict[demo_selection]
    current_demo.display()

    #raccoon = matl.Image.load('resources/images/raccoon.jpg')
    #raccoon.compute(lambda arr: arr * (x / 100)).display()


if __name__ == '__main__':
    main()
