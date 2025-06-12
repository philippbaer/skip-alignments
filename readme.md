# Skip Alignment Computation
<sub><sup>Philipp Bär, Moe T. Wynn, and Sander J. J. Leemans</sup></sub>

In process mining, alignments are a core concept to synchronize actual process executions with a process model.
This repository contains the code to compute _skip alignments_ for a given trace and SBWF-net. The nets are given by their hierarchy, i.e., we represent them with process trees. We further provide the code to run the evaluation that is discussed in the paper "A Full Picture in Conformance Checking: Efficiently Summarizing All Optimal Alignments".

## Structure of This Repository
This repository contains everything needed to compute skip alignments and to recreate the evaluation from the paper. To compare the results with non-skip alignments, you specifically need the PM4py package.

Run
```
pip install -r requirements.txt
```
to install all needed libraries.

You can recreate the evaluation results with `im_models.ipynb`, `indulpet_models.ipynb`, and `rand_models.ipynb`. This might take a few days.

The .py files carry the algorithms to compute skip alignments, and they box the PM4py implementation.

## Running Example
An introduction to compute optimal alignments and skip alignments in normal form is given in `examples.ipynb`. We discuss
- the running example from the paper (example 1)
- a process model with $tau$-loops (example 2)

## Required Event Logs
You need to download the event logs used in this repository to recreate the evaluation results. Download, extract, and save the .xes files to disk. You need to provide the paths to these files in each notebook.

- Road Fines: [Download](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5)
- Request For Payment: [Download](https://doi.org/10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030)
- Sepsis: [Download](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460)

## Precomputed Results
We provide the computational results used in our evaluation in the folders `im_results`, `indulpet_results`, and `rand_results` in `artifacts.zip`. They are equivalent to the files obtained by running the three notebooks again.

## Third Party Dependencies
As scientific library in the Python ecosystem, we rely on external libraries to offer our features. We refer to [this](https://github.com/process-intelligence-solutions/pm4py/tree/release/third_party) page for a detailed list of licenses of the dependencies used in this project. We specifically modified the PM4py library to perform our computations and refer to the PM4py [license](https://github.com/process-intelligence-solutions/pm4py/blob/release/LICENSE) for details.