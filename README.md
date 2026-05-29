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

Scree plots can be useful to 
track the variance decay along principal components,
choose the amount of principal components to keep,
and interpret the finding of PCA in general
https://en.wikipedia.org/wiki/Scree_plot

Together with biplot https://en.wikipedia.org/wiki/Scree_plot 
to see data dispersion along principal components


# Classification using Machine Learning + PCA

PCA can be used to simplify a dataset 
before feeding it into a machine learning model.

## Breast cancer wisconsin DataSet



# Classification using Raw PCA

(See references) Raw PCA can be used to Classification by 
computing the principal components of each class, 
then we may assume a new datapoint projected into each Class Principal Componets Basis
will have small representation error for the class it belongs,
and a big representation error for the classes it does not belong.

## Breast Cancer wisconsin DataSet

32 numerical features. Each register is labeled in Benign or Malignum

### Principal Components decomposition of each class

![](assets/cumulative_variance_raw_pca_breast_cancer_wisconsin.png)

This features are incredibilly highlly correlated. 
A single linear combination (resulting in a single scalar!) 
of the features of a samples is enough to 
represent more than 99% of the data variance

The cumulative explained variance of the classes show that 
Benign tumours are relatively slightly more correlated 
than Malignous ones. 

Although both classes show to be very representable in few components, we may expect some fake negatives for Malignum, 
since its **relatively** harder for _Malign_ to be resumed by less components (In comparisson to _Benign_). 
I.e. I expect the majority of errors from the model to come from
Malignous samples being classified as Benign ones.

Raw PCA classification main thesis can be previewd by checking (un)similarity 
measurements dimension-wise between the found principal components
for each class.

![](assets/PCs_cosine_distances_breast_cancer_wisconsin.png)
OOPS ! Both classes have THE SAME principal component 0 (99.9% of data variability). 
So there will be no reconstruction error relative to PC1. 
We'll need to change approachs here.

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

Check the section on [Breast cancer wisconsin with modified pca](#classification-using-raw-modified-pca) 

# Classification using Raw Modified PCA

# Breast cancer wisconsin

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
