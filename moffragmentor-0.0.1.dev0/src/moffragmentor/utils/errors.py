# -*- coding: utf-8 -*-
"""Errors reused across the moffragmentor package"""
__all__ = ["JavaNotFoundError", "NoMetalError"]


class JavaNotFoundError(ValueError):
    """Raised if Java executable could not be found"""


class NoMetalError(ValueError):
    """Raised if structure contains no metal"""
