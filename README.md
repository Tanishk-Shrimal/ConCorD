# ConCorD

ConCorD computes lattice QCD correlation functions and correlation matrices in distillation (LapH).
You write the hadron operators in Python. ConCorD performs the Wick contractions, finds the tensors
and contraction steps shared across the whole correlation matrix, and saves the result as a plan.
The plan runs on your Laplacian eigenvectors and perambulators, on a CPU node or a GPU, and writes
the correlators to an HDF5 file.

ConCorD does not compute eigenvectors or perambulators; it contracts precomputed ones.

## Installation

ConCorD needs Python 3.11 or newer. From the repository root:

```bash
python -m pip install .
concord --help
```

For GPU runs, install a CUDA build of PyTorch first.

## Quick start

The examples run on random data, so the numbers mean nothing; they show the workflow.

```bash
python tools/make_random_laph_data.py --output-dir examples/data
cd examples/pion
concord plan job.yaml          # Wick contractions and planning; writes plan.concord
concord inspect plan.concord   # terms, FLOPs and memory of the plan; reads no lattice data
concord run job.yaml           # contracts the data; writes output_pion.h5
```

`examples/proton/` runs the same way. `examples/displaced_operators/` defines operators with
covariant derivatives and gauge transports, and `examples/concord_quickstart.ipynb` uses the Python
API without a job file.

## Input data

- Laplacian eigenvectors and perambulators as HDF5 datasets; the job file names their axes.
- Gauge links, needed only for covariant derivatives and gauge transports: smeared-gauge files
  written by chroma_laph or QUDA-LapH, or files in the openQCD format (CLS configurations). ConCorD
  uses the links as stored and does not smear them.

## Documentation

- [`docs/user_guide.md`](docs/user_guide.md): writing and running jobs, the conventions, what has
  been checked against other codes, and the current limitations.
- [Inside ConCorD](https://tanishk-shrimal.github.io/ConCorD/inside_concord.html): a step-by-step
  explanation of what happens between the operators and the correlation matrix (the page is
  `docs/inside_concord.html`).

## Tests

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## License and citation

ConCorD is released under the MIT License. If you use it in published work, please cite it as given
in [`CITATION.cff`](CITATION.cff).
