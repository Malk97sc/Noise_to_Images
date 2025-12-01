# Package Desing

This package was designed to keep the project clean, modular, and easy to extend. Each component has a purpose, making it simple to experiment.

## Structure

```bash
src/
├── diffusion/
│   ├── __init__.py
│   ├── schedules.py
│   ├── forward.py
│   ├── models/
│   ├── utils/
│   └── README.md
```

## Local Installation

```bash
pip install -e .
```

## Use

``` python
import diffusion
```