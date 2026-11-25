# ConCorD user guide

This guide explains how to write, compile and run ConCorD jobs, the conventions ConCorD uses, and
what has been checked. The examples in `examples/` run on random data written by
`tools/make_random_laph_data.py` (see the README).

## 1. The workflow

A job is an operator file (Python) and a job file (YAML):

```text
operators.py + job.yaml
   concord plan job.yaml            -> plan.concord   (Wick contractions and planning)
   concord inspect plan.concord        costs and memory; reads no lattice data
   concord run job.yaml --dry-run      checks the work and the input files
   concord run job.yaml             -> results.h5      (+ optional HTML report)
```

The plan is compiled once and reused for every configuration. `concord run` compiles it if the
plan file does not exist yet. All commands accept `--help`.

| Command | What it does |
|---|---|
| `concord plan JOB [--force]` | Compiles the job and writes the plan. `--force` replaces an existing plan |
| `concord inspect PLAN [--hardware]` | Summarizes a plan; `--hardware` checks it against the free memory of this machine |
| `concord run JOB [options]` | Runs the job, or resumes it. Options include `--dry-run`, `--replan`, `--device`, `--workers` and `--report` |
| `concord visualize JOB --output FILE` | Writes an HTML report of the operators and the Wick terms without running anything |
| `concord hardware` | Shows the CPUs, memory and GPUs that ConCorD sees |

## 2. Writing the operators

The operator file defines a function, by default `build`, that returns the correlators:

```python
from concord_lqcd.core import Correlator, Operator, dbar, gamma5, u

def build():
    pion = Operator(dbar("x", "c", "s1") * gamma5("s1", "s2") * u("x", "c", "s2"))
    return {"pion": Correlator(pion, pion.dagger())}
```

- The function returns a dictionary of named `Correlator`s, a single `Correlator` (stored as `C0`)
  or a `CorrMatrix` (entries stored as `C[i,j]`). The names are stored as `correlator_id` in the
  output.
- `Correlator(O_sink, O_source)` is a two-point function and `Correlator(O_sink, J, O_source)` a
  three-point function. Operators are listed from sink to source.
- `CorrMatrix([O1, O2, ...])` puts the `O_i` at the sink and their daggers at the source.
  `CorrMatrix(sinks, sources)` lets you give both sides (see section 10).
- Quark fields are `u, d, s, c, b, t, l` and `ubar`, `dbar`, ..., each with a position, a color and
  a spin label. Spin matrices include `gamma5`, `gammaX`, `gammaY`, `gammaZ`, `gammaT`,
  `gamma_identity` and `spin_matrix(name, a, b)`. Color tensors are `epsilon` and `delta`.
- `Operator(expr, momentum=(px, py, pz))` gives the operator a momentum in units of 2 pi / L.
  `dagger()` reverses the momentum. Operators can be added and multiplied by numbers.
- `build_baryon_operator(("u", "d", "u"), diquark_gamma="Cg5", momentum=(0, 0, 0))` builds the
  local baryon `eps_abc (q0_a^T Gamma q1_b) q2_c`; see `examples/proton/operators.py`.
- `derivative(q, "+x")`, `transport(q, "+x", "+y")` and `shift(q, "+x")` displace a quark field in
  space (section 10). Derivatives and transports need gauge links; see
  `examples/displaced_operators/`.
- `operators.parameters` in the job file are passed to the function as keyword arguments.
- The operator file may import Python files from its own directory. ConCorD records them in the
  plan, so editing them means the plan has to be compiled again.

## 3. The job file

`examples/pion/job.yaml` is a complete job. Relative paths are relative to the job file.

```yaml
schema_version: 1
job: {name: example_pion, description: pi+ two-point correlator}
operators: {file: ./operators.py, factory: build}
plan: {file: ./plan.concord}
lattice: {spatial_extent: [4, 4, 4], temporal_extent: 8}
compile:
  mode_size: 8                  # N_v
  spin_basis: DR                # DR (default) or DP; must match the perambulators
  scalar_type: complex128       # or complex64
  contraction: {strategy: greedy}
run:
  device: cpu
  configurations: {values: [1]}
  times:
    generator:
      type: two_point
      source: 0
      sinks: {range: {start: 0, stop: 8}}
data: ...                       # section 3.4
output: {file: ./output_pion.h5, on_existing: overwrite}
```

### 3.1 `compile`: settings stored in the plan

| Option | Default | Meaning |
|---|---|---|
| `mode_size` | required | N_v, the number of eigenvectors used |
| `spin_basis` | `DR` | `DR` (DeGrand-Rossi, as in Chroma) or `DP` (Dirac-Pauli, as in chroma_laph and QUDA-LapH perambulators) |
| `scalar_type` | `complex128` | or `complex64` |
| `contraction.strategy` | `greedy` | How the contraction order is chosen; section 7 |
| `contraction.memory_limit` | none | Largest intermediate tensor allowed, in elements |
| `physical.memory_limit_bytes` | none | Largest total buffer memory of the plan; planning fails if no plan fits |
| `wick_aggregation`, `numerical_flavor_groups` | off | Merging of equal Wick terms; section 7 |

If any of these, the operator file, the function name or its parameters change, the plan has to
be compiled again. ConCorD notices this and stops instead of using an old plan.

### 3.2 `run`: what to compute

`configurations` is a list (`values: [1, 5, 9]`) or a range (`range: {start: 1, stop: 100, step:
4}`, stop excluded). `times` sets the operator times. One configuration with one set of times is a
work unit.

```yaml
times:
  generator:
    type: two_point
    sources: {values: [0, 16, 32, 48]}      # or source: 0
    sinks: {range: {start: 0, stop: 64}}
```

Three-point functions use `type: three_point` with `source`, `sink` and `insertions`. Times can
also be listed one by one:

```yaml
times:
  contexts:
    - {id: t0_t5, operator_times: {2: [5, 0]}}   # two-point: [sink, source]
```

The other `run` options (`device`, `workers`, `threads_per_worker`, `time_batch_size`) are in
section 6. None of them requires a new plan.

### 3.3 `output`

`output.file` is the HDF5 result file. `on_existing` is `resume` (default), `error` or `overwrite`.

### 3.4 `data`: the input files

```yaml
data:
  eigenvectors:
    provider: hdf5_distillation
    path: ../data/eigenvectors.h5            # may contain {configuration} and {time}
    dataset: /eigenvectors                   # may contain {configuration} and {time}
    axes: [time, z, y, x, color, mode]
  propagation:
    provider: hdf5_perambulator
    ordinary:
      files: {u: ../data/perambulator_u.h5, d: ../data/perambulator_d.h5}
      dataset: /perambulators/src_t_{source_time}/data
      axes: [sink_time, source_spin, sink_spin, source_mode, sink_mode]
      source_field: qdagger
  gauge_links:                               # only for derivatives and transports
    provider: laph_iomap                     # or openqcd
    path: ../data/smeared_gauge_cfg{configuration}
```

**Eigenvectors.** Give the axes `x`, `y`, `z`, `color` and `mode` in the order of the file, and
`time` unless the time is in the file or dataset name. Real and imaginary parts may also be stored
on an extra axis called `complex_pair`. The file may hold more than N_v modes; the first N_v are
used. All flavors use the same eigenvectors. The modes must be orthonormal. For local operators at
zero momentum ConCorD uses `V^dagger V = 1` exactly, so eigenvectors that are orthonormal only to
1e-8 give differences of that size from codes that compute `V^dagger V` from the stored vectors.

**Perambulators.** Give one file per flavor with `files`, or one `path` containing `{flavor}`.
Names may contain `{configuration}` and `{flavor}`, and the dataset name also `{source_time}`,
`{sink_time}` or `{fixed_source_time}`. The axes must include `sink_mode`, `sink_spin`,
`source_mode` and `source_spin`, and may include `sink_time`, `source_time` and `complex_pair`. If
the dataset has a `dimension_order` attribute, it is checked against `axes`.

- `source_field: qbar` (default) is for files that store the propagator `<q qbar>`.
  `source_field: qdagger` is for files that store `qbar gamma4` at the source, as QUDA-LapH does.
- `fixed_source_time: t0` is for files that hold a single source time. The opposite time order is
  then obtained as `gamma5 P^dagger gamma5`.
- `compile.spin_basis` must be the basis of the perambulators. ConCorD does not rotate input data.
- Three-point functions with a point current need generalized perambulators.
  `propagation.generalized.resolver: "resolver.py:function"` names a Python function that you
  write and that returns where each such tensor is stored.

**Gauge links** are needed only for derivatives and transports; `shift` needs none. Two formats
are read:

- `provider: laph_iomap`: smeared-gauge files written by chroma_laph or QUDA-LapH;
- `provider: openqcd`: files in the openQCD format, the format of CLS configurations.

ConCorD uses the links as stored and does not smear them. If the derivatives should use the same
links as the eigenvectors, supply the smeared links that were used for the eigenvectors. The path
may contain `{configuration}`. To check an openQCD file, compare
`OpenQCDGaugeReader(path).average_plaquette()` with `.metadata.header_plaquette`
(`from concord_lqcd.job import OpenQCDGaugeReader`).

## 4. Plans

`concord plan job.yaml` writes the plan. Each time a plan is used, ConCorD checks that it matches
the operator file and the `compile` settings. If it does not, the run stops; use
`concord plan job.yaml --force` or `concord run job.yaml --replan`. Data paths, configurations,
times and the device are not part of the plan.

`concord inspect plan.concord` shows, without reading lattice data, the number of Wick terms, the
number of distinct contraction steps, the estimated FLOPs of one work unit and the memory the plan
needs. These are estimates; to compare speeds, run a few time slices with `--report` (section 9).

## 5. Running jobs and reading the results

`concord run job.yaml --dry-run` checks every work unit and input file without computing. Then
`concord run job.yaml` runs the job. Results are written as soon as they are computed. With
`on_existing: resume` (the default), running again skips the finished work units. ConCorD refuses
to resume if the plan, the lattice, the work or the data settings changed.

The result file contains:

| HDF5 object | Content |
|---|---|
| `/correlators/<i>/values` | One correlator, with attribute `correlator_id`. Shape: (configurations, time contexts, open indices, if any) |
| `/work/configurations`, `/work/time_context_ids` | The configurations and the time contexts |
| `/completed` | A done flag per configuration and time context |
| `/metadata` | The plan and the job settings |

```python
import h5py
with h5py.File("output_pion.h5") as f:
    c = f["correlators/0/values"][0]   # first configuration: one value per sink time
```

Correlators are stored at the absolute source and sink times (see section 10 before averaging over
source times).

## 6. CPUs, GPUs and memory

`run.device` (or `concord run --device`) is `cpu` (default), `cuda`, `cuda:N` or `auto`. `auto`
uses a GPU when one is available and the job fits, and the CPU otherwise. If the data fit in host
memory but not on the GPU, a GPU run keeps them in host memory and copies slices to the GPU, which
is slower. One job uses at most one GPU.

On a CPU node, a job that is not very small runs as several worker processes with a few threads
each, because many contraction steps are too small to use many threads well. The split is chosen from the number of
cores, the free memory and the number of work units, and is printed at the start. It can be set
with `run.workers` and `run.threads_per_worker`, or with `--workers` and `--threads-per-worker`.

Several time contexts of one configuration can be contracted together. With
`run.time_batch_size: auto` (default) ConCorD times a few batch sizes once and keeps the fastest;
the result is cached in `~/.cache/concord-lqcd/`. A number fixes the batch size.

ConCorD leaves 20% of the free host memory (10% on a GPU) unused and fits the rest of the work into
the remaining memory. If a run stops because memory runs out, it can be resumed. `concord hardware`
shows the memory ConCorD sees, and `concord inspect plan.concord --hardware` whether a plan fits.

## 7. Contraction strategies and Wick-term merging

`compile.contraction.strategy` chooses the order of the pairwise contractions:

| Strategy | What it does |
|---|---|
| `greedy` (default) | opt_einsum's greedy order for each tensor network |
| `dynamic-programming`, `auto`, `auto-hq` | The opt_einsum searches of the same names |
| `reuse-aware` | Builds a few complete plans with different orders, counts the shared steps in each, and keeps the one with the fewest FLOPs that does not need more memory than the greedy plan (or than `physical.memory_limit_bytes`, if set) |
| `adaptive` | At run time, times the greedy and the reuse-aware plan on the chosen device and runs the faster one |

Wick-term merging is optional. It finds Wick terms that give the same tensor, possibly with a
sign, and computes each such tensor once:

```yaml
compile:
  wick_aggregation:
    enabled: true
  numerical_flavor_groups: [[u, d]]   # optional
```

`numerical_flavor_groups` lists flavors that use the same perambulator files, so that their terms
can be merged too. ConCorD checks that the files are the same before running. `concord inspect`
shows how many Wick terms remain.

## 8. Kernel cache

The tensors built from the eigenvectors (for example `V^dagger exp(ip.x) V` for mesons) can be
saved and reused by later runs:

```yaml
data:
  kernel_cache:
    directory: cache/laph-kernels   # relative to the job file
```

A cached tensor is used only if it was made from the same input files with the same settings;
otherwise it is computed again. Checking the input files costs one full read of each file, so the
cache does not always save time.

## 9. Reports

`concord run job.yaml --report calculation.html` also writes an HTML report and a run record,
`calculation.run.json`. The report shows the operators, the Wick terms as quark-flow diagrams, the
compiled tensor network, the devices used and where the time went. `concord visualize job.yaml
--output calculation.html` writes the report without running the job. Names for the operators in
the diagrams can be given in the job file:

```yaml
visualization:
  operator_labels:
    pion: ["pion sink", "pion source"]   # sink first
```

## 10. Conventions

**Wick contractions.** Quark fields anticommute. The sign of each Wick term is the sign of the
permutation that brings every quark next to the antiquark it is contracted with.

**Perambulators.** ConCorD uses the quark propagator `<q qbar>` projected on the eigenvectors,
`tau(t', t) = V^dagger(t') M^-1(t', t) V(t)`, with the sink at `t'` and the source at `t`.

**Spin basis.** `DR` is the DeGrand-Rossi basis of Chroma, with gamma5 = gamma1 gamma2 gamma3 gamma4
= diag(1, 1, -1, -1). `DP` is the Dirac-Pauli basis, used for the perambulators of chroma_laph and
QUDA-LapH. The basis only decides which gamma matrices go into the plan, so it must be the basis
of the perambulators. Chroma's `Gamma(n)` is `spin_matrix("chroma_n", a, b)`.

**Momentum.** An operator with momentum `p` is `sum_x exp(+i p.x) O(x)`, without a factor 1/V.
`dagger()` changes `p` to `-p`.

**Source and sink.** `CorrMatrix([O])` is the same as `CorrMatrix([O], [O.dagger()])`. For an
operator that is not Hermitian, the choice of the daggered side changes the complex correlator.
To compare with another code, give both sides explicitly and check how that code labels momenta:
a label may refer to the source, the sink or the daggered operator.

**Derivatives and transports.** With gauge links `U`, ConCorD uses

```text
T_{+i} q(x) = U_i(x) q(x + e_i)
T_{-i} q(x) = U_i(x - e_i)^dagger q(x - e_i)
derivative(q, "+i") = (T_{+i} - T_{-i}) q / 2,   derivative(q, "-i") = -derivative(q, "+i")
```

where `e_i` is the unit vector in direction `i`, with periodic boundaries and no lattice-spacing
factor. `transport(q, "+x", "+y")` is `U_x(x) U_y(x + e_x) q(x + e_x + e_y)`: the path is read from
the field position outward. Nested
operations apply from the inside out, so `derivative(transport(q, "+y"), "+x")` transports first.
Time derivatives, transports and shifts are not supported.

**Source-time averaging.** Correlators are stored at absolute times. When you shift several source
times to a common origin, a correlator that wraps around the time boundary picks up the boundary
sign of the quarks: -1 for a baryon with antiperiodic quarks, +1 for a meson. ConCorD does not
apply this sign.

**Normalization.** ConCorD applies no volume factors. For the local four-quark operators we
compared, Chroma's results are `V^2` times ConCorD's, with `V` the number of spatial sites. The
bilocal two-meson operators agree without such a factor.

## 11. Validation and limitations

The complete complex correlators were compared with Chroma, chroma_laph, independent NumPy
contractions and exact flavor identities. The largest relative deviations:

| Check | Compared with | Deviation |
|---|---|---|
| Pseudoscalar and vector mesons, at rest and moving | Chroma | 6.0e-15 (3.4e-9 at rest, see below) |
| Bilocal two-meson operators, 13 channels | Chroma | 1.0e-14 |
| Local four-quark operators, 13 channels (after the factor `V^2`) | Chroma | 3.2e-14 |
| Pions with derivatives or transports; proton with a transported quark | chroma_laph | 6.7e-15 |
| Delta, p and Lambda | NumPy | 4.0e-15 |
| p pi+, N pi, N sigma, N pi pi | NumPy | 5.1e-15 |
| Two nucleons (1S0, 3S1) and the triton | NumPy | 1.2e-14 |
| Scalar and gamma4 quark loops at rest; scalar loop at `p = (1,0,0)` | NumPy | 4.0e-15 |
| Delta+ = Delta++ and Lambda = p for equal perambulators; NN isospin relations | exact | 7.5e-16 |
| Three-point function with a point current | full propagators, synthetic data | 2.5e-14 |

The mesons at rest differ from Chroma by 3.4e-9 because ConCorD uses `V^dagger V = 1`, while the
stored eigenvectors are orthonormal only to that precision (section 3.4). The lattice data of the
Chroma and chroma_laph comparisons are not part of the repository. The NumPy checks are in `tools/`
and run on random data as part of the tests.

Current limitations:

- Derivatives and transports were checked for the pion channels and the proton above. Other spin
  structures, displaced baryon quarks and longer paths are implemented but not checked against
  another code.
- Quark loops were checked only as listed above. Loops with derivatives, displacements or
  stochastic estimators are not validated. Vacuum subtraction is left to the analysis.
- Three-point functions with a point current were checked on synthetic data only.
- One job runs on one node, with CPU worker processes or one GPU. There is no MPI.
- For operators with many quarks at one point, memory grows quickly with N_v. `concord inspect`
  shows the memory a plan needs before any data are read.
