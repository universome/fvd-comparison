from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import numpy as np
import torch

import our_fvd
import their_fvd
from util import open_url


def compute_their_fvd(videos_fake: np.ndarray, videos_real: np.ndarray) -> float:
    with tf.Graph().as_default():
        videos_fake = tf.convert_to_tensor(videos_fake, np.float32)
        videos_real = tf.convert_to_tensor(videos_real, np.float32)

        result = their_fvd.calculate_fvd(
            their_fvd.create_id3_embedding(videos_fake),
            their_fvd.create_id3_embedding(videos_real)
        )

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            return sess.run(result)


@torch.no_grad()
def compute_our_fvd(videos_fake: np.ndarray, videos_real: np.ndarray, device: str='cuda') -> float:
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.

    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()

    return our_fvd.compute_fvd(feats_fake, feats_real)


def main():
    seed_fake = 1
    seed_real = 2
    num_videos = 128
    video_len = 16

    videos_fake = np.random.RandomState(seed_fake).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)
    videos_real = np.random.RandomState(seed_real).rand(num_videos, video_len, 224, 224, 3).astype(np.float32)

    print('<==============================================================>')
    print('Computing our FVD...')
    print('<==============================================================>')
    our_fvd_result = compute_our_fvd(videos_fake, videos_real)

    print('<==============================================================>')
    print('Computing their FVD...')
    print('<==============================================================>')
    their_fvd_result = compute_their_fvd(videos_fake, videos_real)

    print('<==============================================================>')
    print(f'[FVD scores] Theirs: {their_fvd_result}. Ours: {our_fvd_result}')
    print('<==============================================================>')


if __name__ == "__main__":
    main()
