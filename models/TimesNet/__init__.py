"""
TimesNet module initialization.

This file exposes the public API for the TimesNet package by importing
the main model and configuration classes.

Exports
-------
TimesNetModel
    Forecasting model implementation.
TimesNetConfig
    Hyperparameter container for TimesNetModel.
"""

from .model import TimesNetModel, TimesNetConfig

__all__ = ["TimesNetModel", "TimesNetConfig"]