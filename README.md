<div align="center">

# Meta Flow Matching

[![Paper](http://img.shields.io/badge/paper-arxiv.2408.14608-B31B1B.svg)](https://arxiv.org/abs/2408.14608)
[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.9+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/lazaratan/meta-flow-matching/blob/main/LICENSE)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

<div align="center">

<p float="left">
  <img align="top" align="middle" width="40%" src="assets/gif_mfm_letters_train_50.gif" style="display: inline-block; margin: 0 auto; max-width: 300px"/>
  <img align="top" align="middle" width="40%" src="assets/gif_mfm_letters_test_50.gif" style="display: inline-block; margin: 0 auto; max-width: 300px" />
</p>


<div align="left">

## Description

<div align="left">

**Meta Flow Matching (MFM)** is a practical approach to integrating along vector fields on the Wasserstein manifold by amortizing the _flow_ model over the initial distributions. Current flow-based models are limited to a single initial distribution/population and a set of predefined conditions which describe different dynamics.

In natural sciences, multiple processes can be represented as vector fields on the Wasserstein manifold of probability densities - i.e. the change of the population at any moment in time depends on the population itself due to the interactions between samples/particles. One domain of applications is personalized medicine, where the development of diseases and the respective effect/response of treatments depend on the microenvironment of cells specific to each patient.

In MFM, we jointly train a vector field model $v_t(\cdot | \varphi(p_0; \theta); \omega)$ and a population embedding model $\varphi(p_0; \theta)$. Initial populations are embedded into lower dimensional representations using a Graph Neural Network (GNN). This gives MFM the ability to generalize over unseen distributions, unlike previously proposed methods. We show the ability of MFM to improve prediction of individual treatment responses on a [large-scale multi-patient single-cell drug screen dataset](https://www.cell.com/cell/pdf/S0092-8674(23)01220-5.pdf).

This repo contains all elements needed to reproduce our results. See [this http link](https://arxiv.org/abs/2408.14608) for the paper.

The data can be downloaded here: [Data (biological data download link)](https://data.mendeley.com/datasets/hc8gxwks3p/1)

<div align="left">
  
If you find this code useful in your research, please cite the following paper (expand for BibTeX):


<details>
<summary>
L. Atanackovic*, X. Zhang*, B. Amos, M. Blanchette, L.J. Lee, Y. Bengio, A. Tong, K. Neklyudov. Meta Flow Matching: Integrating Vector Fields on the Wasserstein Manifold, 2024.
</summary>

```bibtex
@article{atanackovic2024meta,
      title={Meta Flow Matching: Integrating Vector Fields on the Wasserstein Manifold}, 
      author={Lazar Atanackovic and Xi Zhang and Brandon Amos and Mathieu Blanchette and Leo J. Lee and Yoshua Bengio and Alexander Tong and Kirill Neklyudov},
      year={2024},
      eprint={2408.14608},
      archivePrefix={arXiv},
}
```

</details>

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/lazaratan/meta-flow-matching.git
cd meta-flow-matching

# [OPTIONAL] create conda environment
conda create -n mfm python=3.9
conda activate mfm

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with chosen experiment configuration from [src.conf/experiment/](src/conf/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py experiment=experiment_name.yaml trainer.max_epochs=1234 seed=42
```

To train a model via MFM on the synthetic letters setting, use

```bash
python train.py experiment=letters_mfm.yaml
```

To run the biological experiments, first download and pre-process the data using the [trellis_data.ipynb](notebooks/trellis_data.ipynb). Then, similar to the synthetic letters experiment, executing

```bash
python train.py experiment=trellis_mfm.yaml
```

will train 1 seed of an MFM model on the organoid drug-screen dataset.

To replicate an experiment, for example, the last row of Table 1 (in the paper), you can use the multi-run feature:

```bash
python train.py -m experiment=letters_mfm.yaml seed=1,2,3
```

</div>

## Contributions
<div align="left">

Have a question? Found a bug? Missing a specific feature? Feel free to file a new issue, discussion or PR with respective title and description.

Before making an issue, please verify that:

- The problem still exists on the current `main` branch.
- Your python dependencies are updated to recent versions.

Suggestions for improvements are always welcome!
