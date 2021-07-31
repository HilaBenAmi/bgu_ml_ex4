# Self-ensembling for visual domain adaptation (small images) - BGU Project

A final project in ML course in Ben Gurion University. 
The project based on the paper [Self-ensembling for visual domain adaptation](https://arxiv.org/abs/1706.05208),
accepted as a poster at ICLR 2018.

The code was adapted from 
[https://github.com/Britefury/self-ensemble-visual-domain-adapt-photo/](https://github.com/Britefury/self-ensemble-visual-domain-adapt-photo/).


## Installation

You will need:

- Python 3.8 (Anaconda Python recommended)
- OpenCV with Python bindings
- PyTorch

First, install OpenCV and PyTorch as `pip` may have trouble with these.

### OpenCV with Python bindings

On Linux, install using `conda`:

```> conda install opencv```

On Windows, go to [https://www.lfd.uci.edu/~gohlke/pythonlibs/](https://www.lfd.uci.edu/~gohlke/pythonlibs/) and
download the OpenCV wheel file and install with:

```> pip install <path_of_opencv_file>```

### PyTorch

For installation instructions head over to the [PyTorch](https://pytorch.org/) website.

### The rest

Use pip like so:

```> pip install -r requirements.txt```

## Usage

Domain adaptation experiments are run via the `experiment_domainadapt_meanteacher.py`.
Supervised experiments are run via the `sup_experiment.py`.

The experiments in my project can be re-created by execute the `run_experiments.bat`.

There are also notebooks which are set to run on GPU.

## Syn-Digits datasets

You will need to download the Syn-Digits dataset. After this you will need to create
a folder name `data` in which you will store this dataset.

### Syn-digits

Download Syn-digits from [http://yaroslav.ganin.net](http://yaroslav.ganin.net), on which you will find a Google Drive
link to a file called `SynthDigits.zip`. Create a directory call `syndigits` within `data` and unzip `SynthDigits.zip`
within it.
