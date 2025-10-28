## Alignment of spatial transcriptomics slices across diseases, platforms and conditions

###  Yu Wang, Zaiyi Liu, Qingchen Zhang, Xiaoke Ma

AlignDG is the network-based model for aligning distant slices without prior information, covering scenarios such as non-consecutive, non-continuous, across different tissues and diseases conditions that cannot be properly handled with available methods. Furthermore, AlignDG joints feature learning and slice alignment, where features are learned under the guidance of alignment, thereby improving discriminative and quality of features. Extensive experimental results demonstrate the superiority of AlignDG over existing state-of-the-art in terms of precision, robustness and efficiency. AlignDG precisely tracks development of tissues from spatial-temporal slices by only using approximately 50% slices in datasets, which provides biologists with a new perspective to design experiments and analyze spatial transcriptomics data.

![AlignDG workflow](https://raw.githubusercontent.com/xkmaxidian/AlignDG/master/docs/AlignDG.png)

# Installation

##### (Note: To accelerate AlignDG by using GPU: If you have an NVIDIA GPU and using Linux OS, please install Pytorch and jaxlib-cuda in previous, the CPU version of them will be installed by default for you. Here is the [installation guide of PyTorch](https://pytorch.org/get-started/locally/) and the [installation guide of jaxlib](https://jax.readthedocs.io/en/latest/installation.html))

### The detailed tutorials for install AlginDG is available at: https://aligndg-tutorials.readthedocs.io/en/latest/Installation.html

#### 1. Start by using python virtual environment with [conda](https://anaconda.org/):

```
conda create --name aligndg_py python=3.10
conda activate aligndg_py
```

Note: If you encounter the error message "ImportError: Please install the skmisc package via `pip install --user scikit-misc`" while executing `sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=3000)`, please execute the following command in your terminal: `pip install --user scikit-misc`.

Then, you can install AlignDG via Pypi:

```python
pip install aligndg
```



### 2. From source code (Github)

```
git clone https://github.com/xkmaxidian/AlignDG.git
cd your_dir/AlignDG/AlignDG_package
python setup.py build
python setup.py install
```



## Tutorial

1. Tutorial documentation for AlignDG  generated using `ReadTheDocs`is accessible from: https://aligndg-tutorials.readthedocs.io/en.

2. The tutorials for AlignDG source code are summarized as:

The jupyter Notebooks of the tutorial for the balanced simulate data (data 1 and 2) are also accessible from : 

https://github.com/xkmaxidian/AlignDG/blob/master/Tutorials/Tutorials_Simulate1_Banalced.ipynb

https://github.com/xkmaxidian/AlignDG/blob/master/Tutorials/Tutorials_Simulate2_Balanced.ipynb

<br>

The jupyter Notebook of the tutorial for the unbalanced simulate data is accessible from : 
https://github.com/xkmaxidian/AlignDG/blob/master/Tutorials/Tutorials_simulate.ipynb

<br>

The tutorials for AlignDG on real spatial transcriptomics are accessible from:

* For DLPFC: https://github.com/xkmaxidian/AlignDG/blob/master/Tutorials/Tutorials_DLPFC_Align.ipynb
* For MERFISH: https://github.com/xkmaxidian/AlignDG/blob/master/Tutorials/Tutorials_MERFISH_Align.ipynb

<br>

Tutorials for AlignDG on multiple slices and 3D reconstruction are accessible from:

* For AlignDG (Pairwise): https://github.com/xkmaxidian/AlignDG/blob/master/Tutorials/Tutorials_DLPFC_pairwise.ipynb
* For AlignDG (Reference): https://github.com/xkmaxidian/AlignDG/blob/master/Tutorials/Tutorials_DLPFC_reference.ipynb



Note: Please install **jupyter notebook** in order to open this notebook.



## Compared slice alignment algorithms in this paper:

Algorithms that are compared include: 

* [PASTE](https://github.com/raphael-group/paste)
* [PASTE2](https://github.com/raphael-group/paste2)
* [STAligner](https://github.com/zhoux85/STAligner)
* [SLAT](https://github.com/gao-lab/SLAT)
* [Moscot](https://github.com/theislab/moscot)
* [CAST](https://github.com/wanglab-broad/CAST)
* [SPACEL](https://github.com/QuKunLab/SPACEL)
* [Spateo](https://github.com/aristoteleo/spateo-release)
* [DeepST](https://github.com/JiangBioLab/DeepST)
* [GraphST](https://github.com/JinmiaoChenLab/GraphST)
* [Seurat](https://github.com/satijalab/seurat)
* [Harmony](https://github.com/immunogenomics/harmony)

We also acknowledge the authors who provided source code and data which inspire us a lot. They include:

* [SEDR](https://github.com/JinmiaoChenLab/SEDR)
* [DeST-OT](https://github.com/raphael-group/DeST_OT)
* [scDOT](https://doi.org/10.1093/bib/bbae072)
* [Graspot](https://github.com/zhan009/Graspot)

### Contact:

We are continuing adding new features. Bug reports or feature requests are welcome.
Last update: 10/20/2025, version 1.0.1

Please send any questions or found bugs to Xiaoke Ma [xkma@xidian.edu.cn](mailto:xkma@xidian.edu.cn).

### Reference

Our paper is under review.

