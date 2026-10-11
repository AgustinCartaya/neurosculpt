"""Instance normalization factory: tensorflow_addons implementation or Keras ``GroupNormalization(groups=-1)``.

The original variants differ only in which implementation they used:

* ``"tfa"``: ``tensorflow_addons.layers.InstanceNormalization`` (MGAN_paper_16, MGAN_paper_16_seg_loss,
  MPGAN_paper_16). Requires TF <= 2.15 with ``tensorflow_addons`` installed.
* ``"group"``: ``tf.keras.layers.GroupNormalization(groups=-1)`` (MPGAN_paper_16_seg_loss), which
  computes the same instance normalization and needs no extra dependency (TF >= 2.11).
"""
import tensorflow as tf

NORM_TFA = "tfa"
NORM_GROUP = "group"


def instance_norm(norm, axis=-1, **kwargs):
    """Return an instance-normalization layer normalizing over every axis except batch and ``axis``."""
    if norm == NORM_TFA:
        import tensorflow_addons as tfa
        return tfa.layers.InstanceNormalization(axis=axis, **kwargs)
    if norm == NORM_GROUP:
        return tf.keras.layers.GroupNormalization(groups=-1, axis=axis, **kwargs)
    raise ValueError(f"Unknown normalization '{norm}', expected '{NORM_TFA}' or '{NORM_GROUP}'")
