from setuptools import setup, find_packages
from pathlib import Path

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='medusa-kernel',
    packages=find_packages(),
    version='0.2.2',
    keywords=['Signal', 'Biosignal', 'EEG', 'BCI'],
    url='https://medusabci.com/medusa-kernel',
    author='Eduardo Santamaría-Vázquez, '
           'Víctor Martínez-Cagigal, '
           'Víctor Rodríguez-González',
    author_email='eduardo.santamaria@gib.tel.uva.es',
    license='MIT',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'numba',
        'sklearn',
        'statsmodels',
        'bson',
        'h5py',
        'dill',
        'tqdm'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
    ],
    description='Advanced biosignal processing toolbox',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
