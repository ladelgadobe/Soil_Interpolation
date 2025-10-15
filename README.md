# Soil Interpolation

Repository containing the base scripts developed during my Master’s research for testing and validating soil interpolation methods in Digital Soil Mapping (DSM).

## Overview

This repository provides implementations of classical interpolation methods used to generate soil property maps under different sampling densities and spatial dependence scenarios. The scripts were designed to be modular and reproducible, serving as a foundation for both exploratory analysis and academic outputs.

## Implemented Methods


### Python scripts <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="20"/> 
- Inverse Distance Weighting (IDW) – with power parameter and number of neighbors optimized to minimize prediction error  
- Thin Plate Spline (TPS)

###  R scripts   <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/r/r-original.svg" width="20"/>
- Ordinary Kriging (OK) – fitted with Method of Moments (MoM)  
- Ordinary Kriging (OK) – fitted with Restricted Maximum Likelihood (REML)
- Kriging with External Drift (KED) 
- Random Forest Spatial Interpolation (RFSI) – model that incorporates, as covariates, the values of the n nearest observations and their distances to the prediction location. The idea (according to Sekulić et al., 2020) is that neighboring points carry direct spatial information that can improve prediction.  
- Random Forest Spatial (RFE) – using geographic coordinates (x, y) as covariates  
- Support Vector Machine (SVM)  
- Regression Kriging (RK) – regression base implemented with Random Forest Spatial (RFE)  
