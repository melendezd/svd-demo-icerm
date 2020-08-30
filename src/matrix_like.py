from __future__ import annotations

from nptyping import NDArray, UInt8, Int16, Complex128  # type: ignore
from typing import Any, Callable, TypeVar, Generic

import PIL.Image  # type: ignore
import numpy as np  # type: ignore

import scipy.io.wavfile as wav  # type: ignore
import scipy.signal as sig  # type: ignore

import streamlit as st  # type: ignore

T = TypeVar("T")
S = TypeVar("S")
class IMatrixLike(Generic[T, S]):
    def to_matrix(self) -> NDArray[(Any, Any), T]:
        raise NotImplementedError

    def compute(self, fun: Callable[[NDArray[(Any, Any), T]], NDArray[(Any, Any), T]]) -> S:
        return self.from_matrix(fun(self.to_matrix()))

    def extract(self, fun: Callable[[NDArray[(Any, Any), T]], Any]) -> Any:
        return fun(self.to_matrix())

    @staticmethod
    def from_matrix(mat: NDArray[(Any,  Any), T], **kwargs: Any) -> S:
        raise NotImplementedError


class IDisplayable:
    def display(self) -> None:
        raise NotImplementedError

class ILoadable(Generic[S]):
    @staticmethod
    @st.cache
    def load(fpath: str) -> S:
        raise NotImplementedError


class MatrixLikeDisplayableLoadable(Generic[T, S], IMatrixLike[T, S], ILoadable[S],
                                    IDisplayable):
    pass


class Image(MatrixLikeDisplayableLoadable[UInt8, 'Image']):
    def __init__(self, num_channels: UInt8, img: NDArray[(Any, Any, Any), UInt8]):
        self.img = img
        self.num_channels = num_channels

    def to_matrix(self) -> NDArray[(Any, Any), UInt8]:
        '''
        Flattens an image with any number of channels into a 2D matrix
        '''
        return self.img.reshape(self.img.shape[0], self.img.shape[1] * self.num_channels)

    def compute(self, fun: Callable[[NDArray], NDArray]) -> Image:
        '''
        Applies a function to the flattened version of an image, rounds clips the
        resulting values between 0 and 255, and returns the resulting image
        '''
        fun_img_mat = fun(self.to_matrix())
        fun_img_mat_normalized = np.clip(fun_img_mat, 0, 255).astype(np.uint8)
        return Image.from_matrix(fun_img_mat_normalized, num_channels=self.num_channels)

    @staticmethod
    def from_matrix(mat: NDArray[(Any, Any), UInt8], **kwargs: Any) -> Image:
        '''
        Converts a 2D array to an image given the number of channels as num_channels in kwargs
        '''
        num_channels: UInt8 = kwargs.get('num_channels')
        return Image(num_channels,
                     mat.reshape(mat.shape[0],
                                        mat.shape[1]//num_channels, num_channels))


    def display(self):
        '''
        Displays the image using Streamlit
        '''
        st.image(self.img)

    @staticmethod
    @st.cache
    def load(fpath: str) -> Image:
        img_arr = np.asarray(PIL.Image.open(fpath))
        return Image(img_arr.shape[2], np.asarray(PIL.Image.open(fpath)))


class Audio(MatrixLikeDisplayableLoadable[Complex128, 'Audio']):
    def __init__(self, arr: NDArray, sample_rate: int):
        self.audio = arr
        self.sample_rate = sample_rate 
        if arr.ndim == 1:
            self.num_channels = 1
        else:
            self.num_channels = arr.shape[1]

    def to_matrix(self) -> NDArray[(Any, Any), Complex128]:
        freqs, times, spec = sig.stft(self.audio)
        return spec

    @staticmethod
    def from_matrix(mat: NDArray[(Any, Any), Complex128], **kwargs: Any) -> Audio:
        times, signal = sig.istft(mat)
        return Audio(signal.astype(np.int16), sample_rate=kwargs['sample_rate'])
        #return Audio(signal, sample_rate=kwargs['sample_rate'])

    def compute(self, fun: Callable[[NDArray], NDArray]) -> Audio:
        new_audio = Audio.from_matrix(fun(self.to_matrix()), sample_rate=self.sample_rate)

        return new_audio


    def display(self):
        st.audio(self.encode_wav(self.audio, 16000))

    @st.cache
    def encode_wav(self, arr: NDArray, sample_rate: UInt8) -> bytes:
        if arr.ndim == 1:
            num_channels = 1
        else:
            raise Exception('Stereo not yet supported')

        num_samples = arr.shape[0]
        bits_per_sample = 16
        byte_rate = int(sample_rate * num_channels * bits_per_sample / 8)
        block_align = int(num_channels * bits_per_sample / 8)
        data_size = int(num_samples * num_channels * bits_per_sample / 8)

        raw = bytes()

        # Chunk 1
        raw += b'RIFF'
        raw += int(data_size + 36).to_bytes(4, byteorder='big')
        raw += b'WAVE'

        # Subchunk 1
        raw += b'fmt '
        raw += (16).to_bytes(4, byteorder='little') # PCM
        raw += (1).to_bytes(2, byteorder='little') # Linear quantization
        raw += num_channels.to_bytes(2, byteorder='little')
        raw += sample_rate.to_bytes(4, byteorder='little')
        raw += byte_rate.to_bytes(4, byteorder='little')
        raw += block_align.to_bytes(2, byteorder='little')
        raw += bits_per_sample.to_bytes(2, byteorder='little')

        # Subchunk 2
        raw += b'data'
        raw += data_size.to_bytes(4, byteorder='little')
        #raw += b''.join((int(sample) - 1 + (1<<(bits_per_sample - 1)))\
        #                .to_bytes(bits_per_sample//8, byteorder='little') 
        #                for sample in arr)
        raw += b''.join((int(sample) + (2**14))
                        .to_bytes(bits_per_sample//8, byteorder='little') 
                        for sample in arr)
        #raw += b''.join(int(sample << 1 if sample >= 0 else ~(-sample)+1)
        #                .to_bytes(bits_per_sample//8, byteorder='little') 
        #                for sample in arr)
        return raw


    @staticmethod
    @st.cache
    def load(fpath: str) -> Audio:
        sample_rate, signal = wav.read(fpath)
        return Audio(signal, sample_rate)
