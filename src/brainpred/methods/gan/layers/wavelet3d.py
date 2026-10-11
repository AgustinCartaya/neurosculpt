"""3D discrete wavelet transform (DWT) and its inverse as Keras layers (channels-last volumes)."""
import pywt
import tensorflow as tf
from tensorflow.keras import layers

# The original models were trained with mixed precision and float16 filters
# (OLD layers/Wavelet3D_16.py); ``filter_dtype=tf.float32`` reproduces OLD layers/Wavelet3D.py.
DEFAULT_FILTER_DTYPE = tf.float16


class DWT3D(layers.Layer):
    """Single-level 3D DWT of a ``(batch, x, y, z, 1)`` volume into 8 sub-bands stacked on the channel axis.

    Args:
        wavelet_name: any wavelet name accepted by PyWavelets.
        filter_dtype: dtype of the filter coefficients (must match the layer compute dtype).
    """

    def __init__(self, wavelet_name="haar", filter_dtype=DEFAULT_FILTER_DTYPE, **kwargs):
        super().__init__(**kwargs)
        wavelet = pywt.Wavelet(wavelet_name)
        self.dec_len = wavelet.dec_len

        # Decomposition filters reshaped as 1D convolution kernels along x
        lpf = tf.constant(wavelet.dec_lo[::-1], dtype=filter_dtype)
        self.lpf = tf.reshape(lpf, (self.dec_len, 1, 1, 1, 1))
        hpf = tf.constant(wavelet.dec_hi[::-1], dtype=filter_dtype)
        self.hpf = tf.reshape(hpf, (self.dec_len, 1, 1, 1, 1))

        self.conv_type = "VALID"
        self.border_padd = "SYMMETRIC"
        self.pad = self.dec_len // 2
        self.rem_left = 0
        self.rem_right = 0

    def call(self, inputs, training=None, mask=None):
        conv = lambda x, f: tf.nn.conv3d(x, f, padding=self.conv_type, strides=[1, 1, 1, 1, 1])

        # -------- X axis
        inputs_pad = tf.pad(inputs, [[0, 0], [self.pad - 1, self.pad], [0, 0], [0, 0], [0, 0]], self.border_padd)
        l = conv(inputs_pad, self.lpf)
        h = conv(inputs_pad, self.hpf)
        l = l[:, self.rem_left:l.shape[1] - self.rem_right:2, :, :, :]
        h = h[:, self.rem_left:h.shape[1] - self.rem_right:2, :, :, :]

        # -------- Y axis
        l = tf.pad(l, [[0, 0], [0, 0], [self.pad - 1, self.pad], [0, 0], [0, 0]], self.border_padd)
        h = tf.pad(h, [[0, 0], [0, 0], [self.pad - 1, self.pad], [0, 0], [0, 0]], self.border_padd)
        lpf_y = tf.transpose(self.lpf, perm=[1, 0, 2, 3, 4])
        hpf_y = tf.transpose(self.hpf, perm=[1, 0, 2, 3, 4])
        ll, lh = conv(l, lpf_y), conv(l, hpf_y)
        hl, hh = conv(h, lpf_y), conv(h, hpf_y)
        ll = ll[:, :, self.rem_left:ll.shape[2] - self.rem_right:2, :, :]
        lh = lh[:, :, self.rem_left:lh.shape[2] - self.rem_right:2, :, :]
        hl = hl[:, :, self.rem_left:hl.shape[2] - self.rem_right:2, :, :]
        hh = hh[:, :, self.rem_left:hh.shape[2] - self.rem_right:2, :, :]

        # -------- Z axis
        pad_z = [[0, 0], [0, 0], [0, 0], [self.pad - 1, self.pad], [0, 0]]
        ll, lh = tf.pad(ll, pad_z, self.border_padd), tf.pad(lh, pad_z, self.border_padd)
        hl, hh = tf.pad(hl, pad_z, self.border_padd), tf.pad(hh, pad_z, self.border_padd)
        lpf_z = tf.transpose(self.lpf, perm=[2, 1, 0, 3, 4])
        hpf_z = tf.transpose(self.hpf, perm=[2, 1, 0, 3, 4])

        sub_bands = []
        for x in (ll, lh, hl, hh):
            for f in (lpf_z, hpf_z):
                band = conv(x, f)
                sub_bands.append(band[:, :, :, self.rem_left:band.shape[3] - self.rem_right:2, :])

        # Order: lll, llh, lhl, lhh, hll, hlh, hhl, hhh
        return tf.concat(sub_bands, axis=-1)


class IDWT3D(layers.Layer):
    """Inverse of :class:`DWT3D`: maps 8 stacked sub-bands back to a single-channel volume.

    Args:
        wavelet_name: any wavelet name accepted by PyWavelets.
        filter_dtype: dtype of the filter coefficients (must match the layer compute dtype).
    """

    def __init__(self, wavelet_name="haar", filter_dtype=DEFAULT_FILTER_DTYPE, **kwargs):
        super().__init__(**kwargs)
        wavelet = pywt.Wavelet(wavelet_name)
        self.rec_len = wavelet.rec_len

        # Reconstruction filters reshaped as 1D convolution kernels along x
        lpf = tf.constant(wavelet.rec_lo[::-1], dtype=filter_dtype)
        self.lpf = tf.reshape(lpf, (self.rec_len, 1, 1, 1, 1))
        hpf = tf.constant(wavelet.rec_hi[::-1], dtype=filter_dtype)
        self.hpf = tf.reshape(hpf, (self.rec_len, 1, 1, 1, 1))

        self.pad_type = "VALID"
        self.border_pad = "SYMMETRIC"
        self.pad = self.rec_len // 2

    def call(self, inputs, training=None, mask=None):
        x = tf.pad(inputs,
                   [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [self.pad, self.pad], [0, 0]],
                   self.border_pad)

        # (filter z, filter y, filter x) for each sub-band: lll, llh, lhl, lhh, hll, hlh, hhl, hhh
        lpf, hpf = self.lpf, self.hpf
        filters = [(lpf, lpf, lpf), (hpf, lpf, lpf), (lpf, hpf, lpf), (hpf, hpf, lpf),
                   (lpf, lpf, hpf), (hpf, lpf, hpf), (lpf, hpf, hpf), (hpf, hpf, hpf)]

        bands = []
        for i, (fir1, fir2, fir3) in enumerate(filters):
            band = self.upsampler3d(tf.expand_dims(x[:, :, :, :, i], axis=-1))
            bands.append(self.conv_tr_conv_tr_conv_tr(band, fir1, fir2, fir3, self.pad_type))

        reconstructed = tf.math.add_n(bands)
        # Crop the padded border
        s = reconstructed.shape
        return reconstructed[:, self.pad:s[1] - (self.pad + 1), self.pad:s[2] - (self.pad + 1),
                             self.pad:s[3] - (self.pad + 1), :]

    def upsampler3d(self, x):
        """Zero-insertion upsampling: ``(batch, w, h, d, ch)`` -> ``(batch, 2w, 2h, 2d, ch)``."""
        # ---- Z axis
        stack_slices = tf.stack([x, tf.zeros_like(x)], axis=4)
        stack_slices = tf.reshape(stack_slices, shape=[-1, x.shape[1], x.shape[2], x.shape[3] * 2, x.shape[4]])

        # ---- Y axis
        stack_slices = tf.transpose(stack_slices, perm=[0, 1, 3, 2, 4])
        stack_rows = tf.stack([stack_slices, tf.zeros_like(stack_slices)], axis=4)
        stack_rows = tf.reshape(stack_rows, shape=[-1, x.shape[1], x.shape[2] * 2, x.shape[3] * 2, x.shape[4]])
        stack_rows = tf.transpose(stack_rows, perm=[0, 1, 3, 2, 4])

        # ---- X axis
        stack_rows = tf.transpose(stack_rows, perm=[0, 3, 2, 1, 4])
        stack_cols = tf.stack([stack_rows, tf.zeros_like(stack_rows)], axis=4)
        us_padded = tf.reshape(stack_cols, shape=[-1, x.shape[1] * 2, x.shape[2] * 2, x.shape[3] * 2, x.shape[4]])
        return tf.transpose(us_padded, perm=[0, 3, 2, 1, 4])

    def conv_tr_conv_tr_conv_tr(self, x, fir1, fir2, fir3, pad="VALID"):
        """Separable convolution with ``fir3`` along x, ``fir2`` along y and ``fir1`` along z."""
        x_rec = tf.nn.conv3d(x, fir3, padding=pad, strides=[1, 1, 1, 1, 1])
        fir2 = tf.transpose(fir2, perm=[1, 0, 2, 3, 4])
        x_rec = tf.nn.conv3d(x_rec, fir2, padding=pad, strides=[1, 1, 1, 1, 1])
        fir1 = tf.transpose(fir1, perm=[2, 1, 0, 3, 4])
        return tf.nn.conv3d(x_rec, fir1, padding=pad, strides=[1, 1, 1, 1, 1])
