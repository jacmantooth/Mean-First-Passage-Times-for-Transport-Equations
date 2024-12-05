# Mean-First-Passage-Times-for-Transport-Equations
This repository contains the code and examples used in the research presented in the paper Mean First Passage Times for Transport Equations. You can find the full paper here: https://arxiv.org/abs/2404.00400

## Overview 
The repository includes implementations for the examples discussed in the paper, including:

## Numerical Methods

1. Circular Domain Example. Simulates particle movement along microtubules in a circular domain:
   ![Alt Text](Circledomain/Exitfromadisk/Figs/MFPTSOLcircle.tiff)
2. Square Domain Example: Models the movement of wolves in the wild, incorporating turning kernel biases and preferred movement directions.

   
   <div style="text-align: center;">
  <img src="Wolftrackex/Figs/domiandirectionmmfptSol.png" alt="Circular Domain Example" width="500" />
   </div>

## Physics-Informed Neural Networks (PINNs)

Demonstrated effectiveness in recapturing the solution for the "Exit from the Disk" example:

![Alt Text](Circledomain/Exitfromadisk/Figs/PINNprediction.png)

Full domain predictions using PINNs:

  ![Alt Text](Circledomain/Exitfromadisk/Figs/PINNcircle.png)



# Usage

This repository provides a comprehensive set of tools and scripts for reproducing the numerical simulations and examples detailed in the paper. Each section is accompanied by its corresponding implementation to ensure clarity and reproducibility.

Feel free to explore and modify the code to extend its applications or adapt it to other transport equation scenarios.
