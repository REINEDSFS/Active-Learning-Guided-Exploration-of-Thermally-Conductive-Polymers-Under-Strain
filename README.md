# **Active Learning-Guided Exploration of Thermally Conductive Polymers Under Strain**

This repository contains an open source implementation of the Gaussian Process Regression model and corresponding dataset described in our paper. 

 # **_Abstract_**

Finding amorphous polymers with higher thermal conductivity (TC) is technologically important, as they are ubiquitous in applications where heat transfer is crucial. While TC is generally low in amorphous polymers, it can be enhanced by mechanical strain, which facilitates the alignment of polymer chains. However, using the conventional Edisonian approach, the discovery of polymers that may have high TC after strain can be time-consuming and without the guarantee of success. In this work, we employ an active learning scheme to speed up the discovery of amorphous polymers with high TC under strain. Polymers under 2x strain are simulated using molecular dynamics (MD), and their TCs are calculated using non-equilibrium MD. A Gaussian Process Regression (GPR) model is then built using these MD data as the training set. The GPR model is used to screen the PoLyInfo database, and the predicted mean TC and uncertainty are used towards an acquisition function to recommend new polymers for labeling via Bayesian Optimization. The TC of these selected polymers are then labeled using MD simulations, and the obtained data are incorporated to rebuild the GPR model, initiating a new iteration of the active learning cycle. Over a few cycles, we identified ten strained polymers with significantly higher TC (>1 W/mK) than the original dataset, and the results offer valuable insights into the structural characteristics favorable for achieving high TC of polymers subject to strain.


 # **_Dataset_**

36 polymers are  randomly selected from PoLyInfo to label using MD as the initial dataset. The following figure shows the 36 polymers TC before (blue) and after strain (red). You can find the 36 initail MD-labeled TC in initial training set csv file.

<img width="769" alt="Screenshot 2024-08-17 at 2 22 06 PM" src="https://github.com/user-attachments/assets/d882d8e0-68d5-4cf9-9ca6-09492cd2df8a">


 # **_Note_**

Code and data for academic purpose only.
