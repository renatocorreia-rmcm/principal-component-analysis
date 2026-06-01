<!-- OVERALL DOCUMENTATION STRUCTURE

comparrisions are made between all methods and all datasets

pca
spca
cpca
lda
---
BCW (weak dataset)
FMNIST (strong dataset)

# Principal Component analysis
  What is

  how is used to classification
    why can fail drastically -> include PCs images
  
  Modificated methods that contour original problem
    implementation
    comparission

-->

## Literature review
angles between subspaces can be used as metric for estimate representation error https://www.merl.com/publications/docs/TR2012-058.pdf

* [Linear Discriminant Analysis (NOT PCA)](https://www.nature.com/articles/s43586-024-00346-y)
  
  max between-class distance x min within-class spread.

  explicit designed for classification

* [Supervised PCA]()
  
  retain directions correlated with the target variable.

  performs pca only on features relavant to labels

* [Contrastive PCA](https://arxiv.org/pdf/1709.06716)

  target variance x background variance
    
  maximizes target_variance - a * backg_variance

  [Raman spectroscopy–based biological applications](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/cem.3202)

# Principal Component Analysis in Python

PCA is, above all, a linear mapping. (A projection)

PCA defines a new **orthogonal coordinate system** 
that optimally describes **variance** in a single dataset.

The most variables are correlated to each other,
the most is usefull to apply PCA.

The principal components are often computed by
**eigendecomposition** of the **data covariance matrix** 
or **singular value decomposition** of the **data matrix**.

1. **centralize** and **normalize** data
2. compute **covariance matrix** of data
3. **eigendecompose** covariance matrix of data
4. **normalize** eigenvectors

When in eigenvector basis,
covariance matrix of data 
is a diagonal matrix of each new axis variance 
(each one being directly proportional to eigenvalues) 

# Classification using Raw PCA

(See references) Raw PCA can be used to Classification by 
computing the principal components of each class, 
then we may assume a new datapoint projected into each Class Principal Componets Basis
will have small representation error for the class it belongs,
and a big representation error for the classes it does not belong.

The original authors do not considered similar principal components between classes

I must find a way to ponder, for each principal component of A:
(explained variability in A) x (UNexplained variability in B) 
so i can decide to remove or keep each componet based on that.

NÃO BASTA O PC_i EXPLICAR BEM A VARIABILIDADE DO DATASET DELE,
ELE TAMBÉM PRECISA EXPLICAR MAL A VARIABILIDADE DO OUTRO DATASET

Como medir o quanto um vetor explica a variabilidade de um dataset ?
Como medir a dispersão dos dados na direção de um vetor ?: 
Projeta os pontos nessa reta, pega media dos pontos na reta, e calcula distancia media da media.

Ou seja, ao inves de encontrar a base que melhor expressa o dataset,
podemos encontrar a base que melhor expressa a sua diferença em relação a outro dataset.
SIGNATURE COMPONENTS ANALYSIS

(explained variability in A) / (explained variability in B) 


## Breast Cancer wisconsin DataSet

32 numerical features. Each register is labeled in Benign or Malignum

### Principal Components decomposition of each class


# Classification using Raw Modified PCA

## Breast cancer wisconsin

As seen in [Breast cancer wisconsin with raw pca](#breast-cancer-wisconsin-dataset-1),
this dataset had similar principal components for each class, 
what causes an insignificant reconstruction error in raw pca.

That why will need to modify pca so we find the principal components 
not only that maximize class A explicability, but also
minimize class B explicability


## References
* [_Principal Component Analysis_ (2026) -- Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)
* [_Object detection using image reconstruction with PCA_ (2009) -- Luis Malagón-Borja, Olac Fuentes](https://drive.google.com/file/d/18UFTN1yL34RFQ86GuMTJgNj5UX6Tyjl4/view?usp=sharing)

### Datasets
* https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
* https://www.kaggle.com/datasets/zalando-research/fashionmnist
