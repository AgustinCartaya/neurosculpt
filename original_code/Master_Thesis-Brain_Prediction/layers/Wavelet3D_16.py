import pywt
import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

class DWT3D(layers.Layer):
    """
    Discrete Wavelet transform - tensorflow - keras for 3D volumes
    inputs:
        wavelet_name - wavelet name (from pywavelet library)
        concat - 1 - merge transform output to one channel
               - 0 - split to 8 channels
    """

    def __init__(self, wavelet_name='haar', **kwargs):
        super(DWT3D, self).__init__(**kwargs)
        wavelet = pywt.Wavelet(wavelet_name)
        self.dec_len = wavelet.dec_len

        # Decomposition filter coefficients
        lpf = wavelet.dec_lo
        hpf = wavelet.dec_hi

        # Convert filters into tensors and reshape for convolution
        lpf = tf.constant(lpf[::-1], dtype=tf.float16)
        self.lpf = tf.reshape(lpf, (self.dec_len, 1, 1, 1, 1))

        hpf = tf.constant(hpf[::-1], dtype=tf.float16)
        self.hpf = tf.reshape(hpf, (self.dec_len, 1, 1, 1, 1))

        self.conv_type = "VALID"
        self.border_padd = "SYMMETRIC"

        self.pad = self.dec_len // 2

        self.rem_left = 0
        self.rem_right = 0
    

    def call(self, inputs, training=None, mask=None):
        # -------- X axis
        # Border padding
        inputs_pad = tf.pad(inputs, [[0, 0], [self.pad-1, self.pad], [0, 0], [0, 0], [0, 0]], self.border_padd)
        
        lpf_x = self.lpf
        hpf_x = self.hpf

        # Convolution along x-axis
        l = tf.nn.conv3d(inputs_pad, lpf_x, padding=self.conv_type, strides=[1, 1, 1, 1, 1])
        h = tf.nn.conv3d(inputs_pad, hpf_x, padding=self.conv_type, strides=[1, 1, 1, 1, 1])

        # Downsample along x-axis
        l = l[:, self.rem_left:l.shape[1]-self.rem_right:2, :, :, :]
        h = h[:, self.rem_left:h.shape[1]-self.rem_right:2, :, :, :]

        # -------- Y axis
        # Border padding along y-axis
        l = tf.pad(l, [[0, 0], [0, 0], [self.pad-1, self.pad], [0, 0], [0, 0]], self.border_padd)
        h = tf.pad(h, [[0, 0], [0, 0], [self.pad-1, self.pad], [0, 0], [0, 0]], self.border_padd)

        # Transpose to work with x axis
        lpf_y = tf.transpose(self.lpf, perm=[1, 0, 2, 3, 4])
        hpf_y = tf.transpose(self.hpf, perm=[1, 0, 2, 3, 4])

        # Convolution along y-axis
        ll = tf.nn.conv3d(l, lpf_y, padding=self.conv_type, strides=[1, 1, 1, 1, 1])
        lh = tf.nn.conv3d(l, hpf_y, padding=self.conv_type, strides=[1, 1, 1, 1, 1])
        hl = tf.nn.conv3d(h, lpf_y, padding=self.conv_type, strides=[1, 1, 1, 1, 1])
        hh = tf.nn.conv3d(h, hpf_y, padding=self.conv_type, strides=[1, 1, 1, 1, 1])

        # Downsample along y-axis 
        ll = ll[:, :, self.rem_left:ll.shape[2]-self.rem_right:2, :, :]
        lh = lh[:, :, self.rem_left:lh.shape[2]-self.rem_right:2, :, :]
        hl = hl[:, :, self.rem_left:hl.shape[2]-self.rem_right:2, :, :]
        hh = hh[:, :, self.rem_left:hh.shape[2]-self.rem_right:2, :, :]

        # -------- Z axis
        ll = tf.pad(ll, [[0, 0], [0, 0], [0, 0], [self.pad-1, self.pad], [0, 0]], self.border_padd)
        lh = tf.pad(lh, [[0, 0], [0, 0], [0, 0], [self.pad-1, self.pad], [0, 0]], self.border_padd)
        hl = tf.pad(hl, [[0, 0], [0, 0], [0, 0], [self.pad-1, self.pad], [0, 0]], self.border_padd)
        hh = tf.pad(hh, [[0, 0], [0, 0], [0, 0], [self.pad-1, self.pad], [0, 0]], self.border_padd)

        lpf_z = tf.transpose(self.lpf, perm=[2, 1, 0, 3, 4])
        hpf_z = tf.transpose(self.hpf, perm=[2, 1, 0, 3, 4])

        # Convolution along z-axis
        lll = tf.nn.conv3d(ll, lpf_z, padding=self.conv_type, strides=[1, 1, 1, 1, 1])
        llh = tf.nn.conv3d(ll, hpf_z, padding=self.conv_type, strides=[1, 1, 1, 1, 1])
        lhl = tf.nn.conv3d(lh, lpf_z, padding=self.conv_type, strides=[1, 1, 1, 1, 1])
        lhh = tf.nn.conv3d(lh, hpf_z, padding=self.conv_type, strides=[1, 1, 1, 1, 1])

        hll = tf.nn.conv3d(hl, lpf_z, padding=self.conv_type, strides=[1, 1, 1, 1, 1])
        hlh = tf.nn.conv3d(hl, hpf_z, padding=self.conv_type, strides=[1, 1, 1, 1, 1])
        hhl = tf.nn.conv3d(hh, lpf_z, padding=self.conv_type, strides=[1, 1, 1, 1, 1])
        hhh = tf.nn.conv3d(hh, hpf_z, padding=self.conv_type, strides=[1, 1, 1, 1, 1])

        # Downsample along z-axis 
        lll = lll[:, :, :, self.rem_left:lll.shape[3]-self.rem_right:2, :]
        llh = llh[:, :, :, self.rem_left:llh.shape[3]-self.rem_right:2, :]
        lhl = lhl[:, :, :, self.rem_left:lhl.shape[3]-self.rem_right:2, :]
        lhh = lhh[:, :, :, self.rem_left:lhh.shape[3]-self.rem_right:2, :]

        hll = hll[:, :, :, self.rem_left:hll.shape[3]-self.rem_right:2, :]
        hlh = hlh[:, :, :, self.rem_left:hlh.shape[3]-self.rem_right:2, :]
        hhl = hhl[:, :, :, self.rem_left:hhl.shape[3]-self.rem_right:2, :]
        hhh = hhh[:, :, :, self.rem_left:hhh.shape[3]-self.rem_right:2, :]

        # Concatenate
        x = tf.concat([lll, llh, lhl, lhh, hll, hlh, hhl, hhh], axis=-1)

        return x





class IDWT3D(layers.Layer):
    """
    Inverse Discrete Wavelet transform - tensorflow - keras for 3D volumes
    inputs:
        wavelet_name - wavelet name (from pywavelet library)
        concat - 1 - merge transform output to one channel
               - 0 - split to 8 channels
    """

    def __init__(self, wavelet_name='haar',  **kwargs):
        super(IDWT3D, self).__init__(**kwargs)
        wavelet = pywt.Wavelet(wavelet_name)
        self.rec_len = wavelet.rec_len

        # Decomposition filter coefficients
        lpf = wavelet.rec_lo
        hpf = wavelet.rec_hi

        # Convert filters into tensors and reshape for convolution
        lpf = tf.constant(lpf[::-1], dtype=tf.float16)
        self.lpf = tf.reshape(lpf, (self.rec_len, 1, 1, 1, 1))

        hpf = tf.constant(hpf[::-1], dtype=tf.float16)
        self.hpf = tf.reshape(hpf, (self.rec_len, 1, 1, 1, 1))

        self.pad_type = "VALID"
        self.border_pad = "SYMMETRIC"

        self.pad = self.rec_len // 2


    def call(self, inputs, training=None, mask=None):
        x = tf.pad(inputs,
            [[0, 0], [self.pad, self.pad], [self.pad, self.pad], [self.pad, self.pad], [0, 0]],
            self.border_pad)
        
        # split again with border padding
        lll = tf.expand_dims(x[:,:,:,:,0], axis = -1)
        llh = tf.expand_dims(x[:,:,:,:,1], axis = -1)
        lhl = tf.expand_dims(x[:,:,:,:,2], axis = -1)
        lhh = tf.expand_dims(x[:,:,:,:,3], axis = -1)

        hll = tf.expand_dims(x[:,:,:,:,4], axis = -1)
        hlh = tf.expand_dims(x[:,:,:,:,5], axis = -1)
        hhl = tf.expand_dims(x[:,:,:,:,6], axis = -1)
        hhh = tf.expand_dims(x[:,:,:,:,7], axis = -1)

        lll = self.upsampler3d(lll)
        llh = self.upsampler3d(llh)
        lhl = self.upsampler3d(lhl)
        lhh = self.upsampler3d(lhh)

        hll = self.upsampler3d(hll)
        hlh = self.upsampler3d(hlh)
        hhl = self.upsampler3d(hhl)
        hhh = self.upsampler3d(hhh)

        lll = self.conv_tr_conv_tr_conv_tr(lll, self.lpf, self.lpf, self.lpf, self.pad_type)
        llh = self.conv_tr_conv_tr_conv_tr(llh, self.hpf, self.lpf, self.lpf, self.pad_type)
        lhl = self.conv_tr_conv_tr_conv_tr(lhl, self.lpf, self.hpf, self.lpf, self.pad_type)
        lhh = self.conv_tr_conv_tr_conv_tr(lhh, self.hpf, self.hpf, self.lpf, self.pad_type)

        hll = self.conv_tr_conv_tr_conv_tr(hll, self.lpf, self.lpf, self.hpf, self.pad_type)
        hlh = self.conv_tr_conv_tr_conv_tr(hlh, self.hpf, self.lpf, self.hpf, self.pad_type)
        hhl = self.conv_tr_conv_tr_conv_tr(hhl, self.lpf, self.hpf, self.hpf, self.pad_type)
        hhh = self.conv_tr_conv_tr_conv_tr(hhh, self.hpf, self.hpf, self.hpf, self.pad_type)

        # add all together
        reconstructed = tf.math.add_n([lll, llh, lhl, lhh, hll, hlh, hhl, hhh])
        # crop the paded part
        rec_shape = reconstructed.shape
        return reconstructed[:, self.pad:rec_shape[1]-(self.pad+1), self.pad:rec_shape[2]-(self.pad+1), self.pad:rec_shape[3]-(self.pad+1), :]
    


    def upsampler3d(self, x):
        """
        Up sampling with zero insertion between the slices, rows, and columns.
        :param x: 5 dim tensor (batch_size, w, h, d, ch)
        :return:  up sampled tensor with shape (batch_size, 2*w, 2*h, 2*d, ch)
        """

        # ---- upsample Z axis
        # Create zero tensor of the same shape
        zero_tensor = tf.zeros_like(x)
        # Stack tensors along the channel axis
        stack_slices = tf.stack([x, zero_tensor], axis=4)
        # Reshape for zero insertion between the slices
        stack_slices = tf.reshape(stack_slices, shape=[-1, x.shape[1], x.shape[2], x.shape[3]*2, x.shape[4]])
        
        # ---- upsample Y axis
        # Transpose in order to insert zeros for the rows
        stack_slices = tf.transpose(stack_slices, perm=[0, 1, 3, 2, 4])
        # Create zero tensor like the padded one
        zero_tensor_1 = tf.zeros_like(stack_slices)
        # Stack tensors along the channel axis
        stack_rows = tf.stack([stack_slices, zero_tensor_1], axis=4)
        # Reshape for zero insertion between the rows
        stack_rows = tf.reshape(stack_rows, shape=[-1, x.shape[1], x.shape[2]*2, x.shape[3]*2, x.shape[4]])
        # Transpose back
        stack_rows = tf.transpose(stack_rows, perm=[0, 1, 3, 2, 4])
        
        # ---- upsample X axis
        # Transpose in order to insert zeros for the columns
        stack_rows = tf.transpose(stack_rows, perm=[0, 3, 2, 1, 4])
        # Create zero tensor like the padded one
        zero_tensor_2 = tf.zeros_like(stack_rows)
        # Stack tensors along the channel axis
        stack_cols = tf.stack([stack_rows, zero_tensor_2], axis=4)
        # Reshape for zero insertion between the columns
        us_padded = tf.reshape(stack_cols, shape=[-1, x.shape[1]*2, x.shape[2]*2, x.shape[3]*2, x.shape[4]])
        # transpose back to normal
        us_padded = tf.transpose(us_padded, perm=[0, 3, 2, 1, 4])
        
        return us_padded


    def conv_tr_conv_tr_conv_tr(self, x, fir1, fir2, fir3, pad="VALID"):
        # -------- X axis
        x_rec = tf.nn.conv3d(x, fir3, padding=pad, strides=[1, 1, 1, 1, 1])

        # -------- Y axis
        fir2 = tf.transpose(fir2, perm=[1, 0, 2, 3, 4])
        x_rec = tf.nn.conv3d(x_rec, fir2, padding=pad, strides=[1, 1, 1, 1, 1])
        
        # -------- Z axis
        fir1 = tf.transpose(fir1, perm=[2, 1, 0, 3, 4])
        x_rec = tf.nn.conv3d(x_rec, fir1, padding=pad, strides=[1, 1, 1, 1, 1])
        
        return x_rec
    