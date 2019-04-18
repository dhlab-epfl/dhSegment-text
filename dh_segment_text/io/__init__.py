r"""
The :mod:`dh_segment.io` module implements input / output functions and classes.

Input functions for ``tf.Estimator``
------------------------------------

**Input function**

.. autosummary::
    input_fn

**Data augmentation**

.. autosummary::
    data_augmentation_fn
    extract_patches_fn
    rotate_crop

**Resizing function**

.. autosummary::
    resize_image
    load_and_resize_image


Tensorflow serving functions
----------------------------

.. autosummary::
    serving_input_filename
    serving_input_image

----

PAGE XML and JSON import / export
---------------------------------

**PAGE classes**

.. autosummary::
    PAGE.Point
    PAGE.Text
    PAGE.Border
    PAGE.TextRegion
    PAGE.TextLine
    PAGE.GraphicRegion
    PAGE.TableRegion
    PAGE.SeparatorRegion
    PAGE.GroupSegment
    PAGE.Metadata
    PAGE.Page

**Abstract classes**

.. autosummary::
    PAGE.BaseElement
    PAGE.Region

**Parsing and helpers**

.. autosummary::
    PAGE.parse_file
    PAGE.json_serialize

----

.. _ref_via:

VGG Image Annotator helpers
---------------------------


**VIA objects**

.. autosummary::
    via.WorkingItem
    via.VIAttribute


**Creating masks with VIA annotations**

.. autosummary::
    via.load_annotation_data
    via.export_annotation_dict
    via.get_annotations_per_file
    via.parse_via_attributes
    via.get_via_attributes
    via.collect_working_items
    via.create_masks


**Formatting in VIA JSON format**

.. autosummary::
    via.create_via_region_from_coordinates
    via.create_via_annotation_single_image

----

"""


_INPUT = [
    'input_fn',
    'serving_input_filename',
    'serving_input_image',
    'data_augmentation_fn',
    'rotate_crop',
    'resize_image',
    'load_and_resize_image',
    'extract_patches_fn',
    'local_entropy'
]

# _PAGE_OBJECTS = [
#     'Point',
#     'Text',
#     'Region',
#     'TextLine',
#     'GraphicRegion',
#     'TextRegion',
#     'TableRegion',
#     'SeparatorRegion',
#     'Border',
#     'Metadata',
#     'GroupSegment',
#     'Page'
# ]
#
# _PAGE_FN = [
#     'parse_file',
#     'json_serialize'
# ]

__all__ = _INPUT  # + _PAGE_OBJECTS + _PAGE_FN

from .input import *
from .input_utils import *
from . import PAGE
from . import via

