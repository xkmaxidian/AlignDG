## Alignment of distant slices of spatially resolved transcriptomics with graph learning model

###  Yu Wang, Zaiyi Liu, Quan Wang, Xiaoke Ma

AlignDG is the first network-based model for aligning distant slices without prior information, covering scenarios such as non-consecutive, non-continuous, across different tissues and diseases conditions that cannot be properly handled with available methods. Furthermore, AlignDG joints feature learning and slice alignment, where features are learned under the guidance of alignment, thereby improving discriminative and quality of features. Extensive experimental results demonstrate the superiority of AlignDG over existing state-of-the-art in terms of precision, robustness and efficiency. AlignDG precisely tracks development of tissues from spatial-temporal slices by only using approximately 50% slices in datasets, which provides biologists with a new perspective to design experiments and analyze SRT data.

![AlignDG workflow](docs/AlignDG.png)

# Installation

#### <font color='red'>To accelerate AlignDG by using GPU: If you have an NVIDIA GPU and using Linux OS, please install Pytorch and jaxlib-cuda in previous, the CPU version of them will be installed by default for you. Here is the [installation guide of PyTorch](https://pytorch.org/get-started/locally/) and the [installation guide of jaxlib](https://jax.readthedocs.io/en/latest/installation.html).</font>

#### 1. Start by using python virtual environment with [conda](https://anaconda.org/):

```
conda create --name aligndg python=3.10
conda activate aligndg
```

(Optional) To run the notebook files in tutorials, please ensure the Jupyter package is installed in your environment:

```
conda install -n aligndg ipykernel
python -m ipykernel install --user --name mnmst --display-name aligndg-jupyter
```

Note: If you encounter the error message "ImportError: Please install the skmisc package via `pip install --user scikit-misc`" while executing `sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)`, please execute the following command in your terminal: `pip install -i https://test.pypi.org/simple/ scikit-misc==0.2.0rc1`.

### 2. From GitHub

```
git clone https://github.com/xkmaxidian/AlignDG.git
```

## Tutorial

A jupyter Notebook of the tutorial for the unbalanced simulate data is accessible from : 
<br>
https://github.com/xkmaxidian/AlignDG/blob/master/Tutorials/Tutorials_simulate.ipynb

Please install **jupyter notebook** in order to open this notebook.

## Compared slice alignment algorithms in this paper:

Algorithms that are compared include: 

* [PASTE](https://github.com/raphael-group/paste)
* [STAligner](https://github.com/zhoux85/STAligner)
* [SLAT](https://github.com/gao-lab/SLAT)
* [moscot](https://github.com/theislab/moscot)
* [Seurat](https://github.com/satijalab/seurat)
* [Harmony](https://github.com/immunogenomics/harmony)

### Contact:

We are continuing adding new features. Bug reports or feature requests are welcome.

Last update: 08/20/2024, version 0.0.1

Please send any questions or found bugs to Xiaoke Ma [xkma@xidian.edu.cn](mailto:xkma@xidian.edu.cn).

### Reference

Our paper is under review.

