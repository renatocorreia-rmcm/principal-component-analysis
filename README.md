# Principal Component Analysis in Python

PCA is, above all, a linear mapping.

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

music dataset https://github.com/mdeff/fma

## References
* [_Principal Component Analysis_ -- Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)