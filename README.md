<p align="center">
<img src="https://raw.githubusercontent.com/socialfoundations/bencheval/main/assets/logo.jpg" height="400" width="600">
</p>

**BenchEval** is a Python package that provides a suite of tools to evaluate multi-task benchmarks focusing on
diversity and sensitivity against irrelevant variations, such as label noise injection and the addition of irrelevant
candidate models. This package facilitates comprehensive analysis of multi-task benchmarks through a social choice lens,
exposing the fundamental trade-off between diversity and stability in both cardinal and ordinal benchmarks.

For more information, including the motivations behind the measures and our empirical findings, please
see [our paper](https://github.com/socialfoundations/bencheval).

## Quick Start

To install the package, simply run:

```bash
pip install bencheval
```

## Example Usage

To evaluate a cardinal benchmark, you can use the following code:

```python
from bencheval.data import load_cardinal_benchmark
from bencheval.measures.cardinal import get_diversity, get_sensitivity

data, cols = load_cardinal_benchmark('GLUE')
diversity = get_diversity(data, cols)
sensitivity = get_sensitivity(data, cols)
```

To evaluate an ordinal benchmark, you can use the following code:

```python
from bencheval.data import load_ordinal_benchmark
from bencheval.measures.ordinal import get_diversity, get_sensitivity

data, cols = load_ordinal_benchmark('HELM-accuracy')
diversity = get_diversity(data, cols)
sensitivity = get_sensitivity(data, cols)
```

To use your own benchmark, you just need to provide a pandas DataFrame a list of columns indicating the tasks.
Check the [documentation](https://socialfoundations.github.io/bencheval) for more details.

## Reproduce the Paper

<p align="center">
<img src="https://raw.githubusercontent.com/socialfoundations/bencheval/main/assets/banner.png" height="400" width="600">
</p>

Please check out [cardinal.ipynb](https://githubtocolab.com/socialfoundations/bencheval/blob/main/examples/cardinal.ipynb), [ordinal.ipynb](https://githubtocolab.com/socialfoundations/bencheval/blob/main/examples/ordinal.ipynb) and [banner.ipynb](https://githubtocolab.com/socialfoundations/bencheval/blob/main/examples/banner.ipynb)
for
reproducing our results.
