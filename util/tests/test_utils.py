import unittest
from unittest import TestCase

import torch

from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from util.misc import _is_numeric, _is_sequence


class TestBoxOps(TestCase):
    def test_xyxy_to_cxcywh(self):
        box = torch.tensor([[2, 2, 2, 4, 4, 4], [3, 3, 3, 6, 6, 6]]).float()

        self.assertTrue(
            torch.allclose(
                box_xyxy_to_cxcywh(box),
                torch.tensor([[3, 3, 3, 2, 2, 2], [4.5, 4.5, 4.5, 3, 3, 3]]),
            )
        )

    def test_cxcywh_to_xyxy(self):
        box = torch.tensor([[3, 3, 3, 2, 2, 2], [4.5, 4.5, 4.5, 3, 3, 3]])

        self.assertTrue(
            torch.allclose(
                box_cxcywh_to_xyxy(box),
                torch.tensor([[2, 2, 2, 4, 4, 4], [3, 3, 3, 6, 6, 6]]).float(),
            )
        )


class TestMisc(TestCase):

    def test_int_is_numeric(self):
        self.assertTrue(_is_numeric(1))

    def test_float_is_numeric(self):        
        self.assertTrue(_is_numeric(1.0))
    
    def test_none_is_not_numeric(self):
        self.assertFalse(_is_numeric(None))

    def test_string_is_not_numeric(self):
        self.assertFalse(_is_numeric("string"))

if __name__ == "__main__":
    unittest.main()
