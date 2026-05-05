# MEDUSA© Kernel

MEDUSA© is a software ecosystem for the development of BCIs and neuroscience experiments. It has two independent components with dfferent goals: MEDUSA© Kernel and MEDUSA© Platform. 

MEDUSA© Kernel is a Python package that contains readyto- use methods to analyze brain signals, including advanced signal processing, machine learning, deep learning, and miscellaneous high-level analyses. It also includes logical functions and classes to handle different biosignals, such as electroencephalography (EEG) and magnetoencephalography (MEG), save experimental data or implement standalone processing pipelines.

## Information

Check the following links to know more about the MEDUSA environment for neurotechnology and brain-computer interface (BCI) experiments:

- Website: [https://www.medusabci.com/](https://www.medusabci.com/)
- Documentation: [https://docs.medusabci.com/kernel/](https://docs.medusabci.com/kernel/)

Important: MEDUSA Kernel is under heavy development! It may change significantly in following versions

## Design philosophy

MEDUSA Kernel adopts a **functional architecture**: signal-processing routines (filters, transforms, metrics, connectivity) are free functions that take their inputs as explicit parameters — a NumPy array, a sampling rate, a few configuration values — and return arrays or scalars. They are not methods bound to a container class.

This is a deliberate departure from libraries such as MNE-Python, whose pipelines are expressed as method chains on a central container (`raw.filter()`, `epochs.average()`, …). The medusa style is lower-level and more verbose at the call site, in exchange for two properties:

- **Transparency.** Every input a function uses appears in its signature. There are no implicit reads from a container's hidden attributes, which makes debugging, unit testing, and partial reuse of the processing chain straightforward.
- **Flexibility.** Functions are independent of any biosignal type. The same filter or metric works on EEG, MEG, ECG, NIRS, or a synthetic test array, because all it sees is an `ndarray` plus the parameters it needs.

Biosignal classes (`EEG`, `ECG`, `EMG`, …) still exist, but their role is **structuring data for persistence and capturing per-modality metadata** (channel sets, montages, reference schemes, sensor coordinates, …) — not dispatching processing methods. They describe *what was recorded*, not *what can be done with it*. This is what lets each modality keep its own rich, faithful schema instead of collapsing into a generic union container.

## Overview
MEDUSA Kernel is a Python library, available in the Python Package Index (PyPI) repository, with a complete suite of functions for signal processing. The included functions can be categorized according to their different levels of abstraction. The first level is composed of low-level functions to process signals and calculate basic parameters, including the following:

- Temporal filters: online and offline infinite impulse response filters (IIR) and offline finite impulse response filters (FIR).
- Spatial filters: common average reference (CAR), laplacian filter, multi-class common spatial patterns (CSP) and canonical correlation analysis (CCA).
- Local activation: including spectral metrics, such as band power, median frequency, Shannon entropy , and complexity metrics, such as central tendency measure, sample entropy, multiscale entropy, Lempel-Ziv's complexity and Multiscale Lempel-Ziv's complexity.
- Connectivity: amplitude metrics, such as amplitude correlation envelope (AEC) and instantaneous amplitude correlation (IAC), and phase metrics, such as phase locking value (PLV), phase lag index (PLI) and weighted PLI (wPLI).

In a higher level of abstraction there are functions that apply a processing pipeline to the input data to analyze certain features. MEDUSA does not assume the nature of the input data in low-level functions, but most of the high-level analysis that are currently implemented are designed to work with electroencephalography (EEG) and magnetoencephalography (MEG) recordings. These functions include:

- Signal processing for BCIs based on event related potentials (ERP): complete classification pipelines including regularized linear discriminant analysis (rLDA), EEGNet and EEG-Inception that can be applied in offline and online modes; ERP analysis with advanced charts.  
- Signal processing for BCIs based on motor imagery (MI): complete classification pipelines including CSP combined with rLDA, EEGNet, EEG-Inception and EEGSym that can be applied in offline and online modes; MI analysis with advanced charts.  
- Signal processing for BCIs based on code-modulated visual evoked potentials (cVEP): complete classification pipeline based on CCA; cVEP analysis with advanced charts. 
- Signal processing for neurofeedback (NF): battery of high-level models based on spectral and connectivity metrics ready to be applied in online and offline applications.

Additionally, the package includes classes and functions to import data from other toolboxes (e.g., MATLAB, MNE), define the data format of signals and experiments, save recordings to several file types (e.g., bson, json, mat) and implement custom real-time signal processing pipelines. Furthermore, some of the functions, including the BCI models, can be applied in both online and offine experiments. Therefore, MEDUSA©  Kernel can be used for offine analysis of previously recorded data, such as public databases, or in real-time tasks. In fact, MEDUSA© Platform relies on this package for signal processing. This is an interesting feature that allows to reproduce the exact same results achieved in an online experiment during subsequent offine analyses, facilitating experimental reproducibility.
