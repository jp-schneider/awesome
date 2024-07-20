# AWESOME


This is the official code repository for:
### *Implicit Representations for Constrained Image Segmentation*
accepted at ICML 2024,

<p align="left">
    <a href="https://openreview.net/pdf?id=IaV6AgrTUp" alt="Paper PDF File">
        <img src="https://img.shields.io/badge/Paper%20PDF-a60a00?logo=adobeacrobatreader" /></a>
    <a href="https://openreview.net/forum?id=IaV6AgrTUp" alt="OpenReview">
        <img src="https://img.shields.io/badge/OpenReview-8c1b13?logo=file" /></a>
    <a href="https://icml.cc/virtual/2024/poster/34423">
        <img src="images/badge/ICML-2024-black.svg"/>
    </a>
    <a href="https://github.com/jp-schneider/awesome/blob/main/images/ICML_poster.pdf" alt="Poster PDF File">
        <img src="https://img.shields.io/badge/Poster%20PDF-a60a00?logo=adobeacrobatreader" /></a>

</p>

<div class="teaser">
    <img src="./images/teaser_combined.png" max_height="400px" max_width="1024px">
</div>


and:

### *Implicit Representations for Image Segmentation*

accepted in the UniReps Workshop at NeurIPS 2023.

<p align="left">
    <a href="https://openreview.net/pdf?id=LSSiDy7fG1" alt="PDF File">
        <img src="https://img.shields.io/badge/Paper%20PDF-a60a00?logo=adobeacrobatreader" /></a>
    <a href="https://openreview.net/forum?id=LSSiDy7fG1" alt="OpenReview Link">
        <img src="https://img.shields.io/badge/OpenReview-8c1b13?logo=file" /></a>
    <a href="https://unireps.org/2023/publication/schneider-2023-implicit/" alt="UniReps Link">
        <img src="https://img.shields.io/badge/UniReps-2023-5d8bc4" /></a>
    <a href="https://github.com/jp-schneider/awesome/blob/main/images/neurips_unireps_poster.pdf" alt="Poster PDF File">
        <img src="https://img.shields.io/badge/Poster%20PDF-a60a00?logo=adobeacrobatreader" /></a>
</p>

<div align="center">
    <img src="./images/teaser_convex.gif" height="400px">
</div>



> AWESOME is a totally serious abbreviation for:
  "Anyone Working on Estimating Segmentations of Objects by Minimizing input-convex Energies".

### TL;DR

In this repository we show how to use shape constraints with *Implicit Representations* to enhance segmentation quality. Is the object, either convex, star-shaped, path-connected, periodic or symmetric in space or time, we show, how this information can be used to regularize any segmentation model or variational approach. One can use our method either as a post-processing step or as a constraint during training.

This can especially be useful if one has not much data at hand, the data is noisy, or existing segmentation models are not accurate or robust.

### Getting Started

To get started, please follow the [Getting Started guide](docs/getting_started.md) to set up the environment.

If you courious how the priors work, we have created short "how-to" notebooks for two of the priors:
- [Convexity](notebooks/how_to/convexity.ipynb)
- [Path-Connectedness](notebooks/how_to/path-connectedness.ipynb)

### Reproducability

Once the environment is set up, we explain in the [reproduction guide](docs/reproduction_guide.md) how to reproduce the results of the paper.

# Execution of the code

The training and evaluation of models can be archieved using seperate configurations and the `run.py` script within the scripts folder.

```bash

python scripts/run.py --config-path [Config-Path]

```

## Citation
If you use our concepts or code in your research, please cite our paper:

```bibtex
@InProceedings{schneider-IRCIS-2024,
    title = {{Implicit} {Representations} for {Constrained} {Image} {Segmentation}},
    author = {Schneider, Jan Philipp and Fatima, Mishal and Lukasik, Jovita and Kolb, Andreas and Keuper, Margret and Moeller, Michael},
    booktitle = {Proceedings of the 41st International Conference on Machine Learning},
    pages = {43765--43790},
    year = {2024},
    volume = {235},
    series = {Proceedings of Machine Learning Research},
    month = {21--27 Jul},
    publisher = {PMLR},
}
```


If you have any doubts or just want to chat about the project, please [contact me](mailto://jan.schneider@uni-siegen.de?subject=AWESOME%20project)!

Best,

Philipp
