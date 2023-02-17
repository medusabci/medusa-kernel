from setuptools import setup, find_packages
from pathlib import Path

# Just testing

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='medusa-kernel',
    packages=find_packages(),
    version='1.1.3',
    keywords=['Signal', 'Biosignal', 'EEG', 'BCI'],
    url='https://medusabci.com/',
    author='Eduardo Santamaría-Vázquez, '
           'Víctor Martínez-Cagigal, '
           'Diego Marcos-Martínez, '
           'Víctor Rodríguez-González, '
           'Sergio Pérez-Velasco',
    author_email='support@medusabci.com',
    license='CC Attribution-NonCommercial-NoDerivs 2.0',
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
        'tensorflow==2.11',
        'tensorflow-probability==0.16'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
    ],
    description='Advanced biosignal processing toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_data={
        'medusa': ['meeg/*.tsv', 'local_activation/*.dll']
    },
    license_files=('LICENSE',),
)
