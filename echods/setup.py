#!/usr/bin/env python3
"""Metadata for package to allow installation with pip."""

import os

import setuptools

setuptools.setup(
    name="echods",
    description="echo dynamic dataset.",
    version='1.0.0',
    url="https://echonet.github.io/dynamic",
    packages=['echods'],
    install_requires=['torchvision', 'torchmetrics', 'torch'],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
