"""
This script compares I3D models used in the original FVD implementation and in StyleGAN-V
"""

import os
import argparse
from typing import Callable

import numpy as np
import torch; torch.set_grad_enabled(False)
import tensorflow as tf

from util import open_url
from their_fvd import create_id3_embedding


def init_tf_model() -> Callable:
    def run_tf_model(x: np.ndarray) -> np.ndarray:
        with tf.Graph().as_default():
            x = tf.convert_to_tensor(x.transpose(0, 2, 3, 4, 1), np.float32)
            embeddings = create_id3_embedding(x)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                embeddings = sess.run(embeddings)

        return embeddings

    return run_tf_model


def init_torch_model(device: str='cpu') -> Callable:
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)

    return lambda x: detector(torch.from_numpy(x), **detector_kwargs).numpy()


if __name__ == '__main__':
    batch_size, c, video_len, h, w = 4, 3, 16, 224, 224
    x = np.random.RandomState(1).rand(batch_size, c, video_len, h, w).astype(np.float32)

    print('<==============================================>')
    print('Initializing models...')
    print('<==============================================>')
    tf_i3d_model = init_tf_model()
    torch_i3d_model = init_torch_model()

    print('<==============================================>')
    print('Computing TensorFlow output...')
    print('<==============================================>')
    y_tf: np.ndarray = tf_i3d_model(x)

    print('<==============================================>')
    print('Computing PyTorch output...')
    print('<==============================================>')
    y_torch: np.ndarray = torch_i3d_model(x)

    # Checking the difference between two outputs
    # On our machine, it gives the output:
    # L_2 difference is 0.00026316816207043225
    print(f'L_2 difference is {((y_tf - y_torch) ** 2).sum() ** 0.5}')
