# =============================================================================
# DEPRECATED — kept only as a fallback for environments that still invoke
# `python setup.py ...` directly. From medusa-kernel 2.0 the canonical
# packaging metadata lives in `pyproject.toml` (PEP 621) and the build is
# driven by `hatchling` via `uv build` / `python -m build`.
#
# This file will be removed once we verify a full release cycle (build +
# publish to PyPI) using only `pyproject.toml`. See TODO K2.
# =============================================================================
import warnings

warnings.warn(
    "medusa-kernel: setup.py is deprecated. Use 'pyproject.toml' "
    "(see TODO K2). This file will be removed in a future 2.0 alpha.",
    DeprecationWarning,
    stacklevel=2,
)

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='medusa-kernel',
    packages=find_packages(),
    version='1.4.3',
    keywords=['Signal', 'Biosignal', 'EEG', 'BCI'],
    url='https://medusabci.com/',
    author='Eduardo Santamaría-Vázquez, '
           'Víctor Martínez-Cagigal, '
           'Diego Marcos-Martínez, '
           'Víctor Rodríguez-González, '
           'Sergio Pérez-Velasco',
    author_email='support@medusabci.com',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'statsmodels',
        'bson',
        'h5py',
        'dill',
        'tqdm',
        'PySide6',
        'PyWavelets'
    ],
    python_requires='>=3.13',
    package_data={
        'medusa': ['meeg/*.tsv', 'signal_metrics/*.dll',
                   'analysis/time_plot/icons/*.png',
                   'analysis/time_plot/time_plot.ui']
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
    ],
    description='Advanced biosignal processing toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='CC Attribution-NonCommercial-NoDerivs 2.0',
    license_files=('LICENSE',),
)
