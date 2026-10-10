# pyaml

> **This is the ALS-APG staging fork of [pyAML](https://github.com/python-accelerator-middle-layer/pyaml),**
> published on PyPI as `accelerator-middle-layer-apg` (import name unchanged: `pyaml`).
> It carries only upstream commits and exists so that OSPREY deployments can depend on
> fixes that are not yet in an upstream release. Unless you deploy OSPREY, install
> `accelerator-middle-layer` instead; do not install both, they ship the same `pyaml/` package.
> Please file issues and pull requests upstream.

**Package for the Python Accelerator Middle Layer core**

[![Documentation Status](https://readthedocs.org/projects/pyaml/badge/?version=stable)](https://pyaml.readthedocs.io/en/stable/?badge=stable)
[![Latest stable release](https://img.shields.io/pypi/v/accelerator-middle-layer.svg)](https://pypi.org/project/accelerator-middle-layer/)

## Overview

<!-- Replace the paragraph below with a longer description of the package. -->

Python Accelerator Middle Layer (pyAML) is a joint technology platform for design, commissioning and operation of particle accelerators. This repository is for the core of pyAML. It is control system independent and is intended to be used together with a package that implements the control system specific interface.

## Installation

Install the package from PyPI:

```bash
pip install accelerator-middle-layer
```

## Development

Install the development dependencies with:

```bash
pip install accelerator-middle-layer[dev]
```

Run the test suite with:

```bash
pytest
```

Install the pre-commit hooks with:

```bash
pre-commit install
```

## Documentation

The documentation is available at:

<https://pyaml.readthedocs.io/en/latest/>

## Contributing

Please use the issue tracker or submit a pull request.
