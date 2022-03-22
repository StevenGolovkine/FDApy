#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Module for the definition for Functional Linear Mixed Models.

This module is used to implements algorithms FLMM. It is used to model the
variance containing within functional data.
"""
import itertools


###############################################################################
# Checkers for parameters


###############################################################################
# Class FLMM


class FLMM():
    """A class defining Functional Linear Mixed Model.

    Parameters
    ----------

    Attributes
    ----------
    """

    def __init__(self):
        """Initialize FLMM object."""

    def fit(self,
            data: FunctionalData
    ):
        """Fit the model on data.

        Parameters
        ----------
        data: FunctionalData
            Training data.
        """
