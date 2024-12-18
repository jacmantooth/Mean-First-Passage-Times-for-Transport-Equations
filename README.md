# Mean-First-Passage-Times-for-Transport-Equations
This repository contains the code and examples used in the research presented in the paper Mean First Passage Times for Transport Equations. You can find the full paper here: https://arxiv.org/abs/2404.00400

## Overview 
This repository provides the numerical implementations and visualizations for the examples discussed in the paper, showcasing applications of Mean First Passage Time (MFPT) theory to various transport equation scenarios. Key examples include:

## Numerical Methods

1. Circular Domain Example.  Simulates particle movement along microtubules within a circular domain.
   <div style="text-align: center;">
    <img src="Circledomain/Figs/g4500120-800px-wm.jpg" 
         alt="Circular Domain Example" 
         width="300" 
         style="display: inline-block; margin-right: 10px;" />
    <img src="Circledomain/Exist_from_both/Figs/MFPTfig.png" 
         alt="Circular Domain Example" 
         width="490" 
         style="display: inline-block;" />
   </div>
2. Square Domain Example:Models the movement of wolves navigating through a domain, accounting for directional preferences such as seismic lines.

   
   <div style="text-align: center;">
    <img src="Wolftrackex/Figs/1000_F_286464561_Kd0xtLPy094435OhOxWnlgNUJeFBF1HP.jpg" 
         alt="Circular Domain Example" 
         width="400" 
         style="display: inline-block; margin-right: 10px;" />
    <img src="Wolftrackex/Figs/domiandirectionmmfptSol.png" 
         alt="Circular Domain Example" 
         width="250" 
         style="display: inline-block;" />
   </div>

## Physics-Informed Neural Networks (PINNs)

We aim to find the Physics-Informed Neural Network (PINN) solution because it offers several advantages over traditional numerical methods like the Finite Element Method (FEM) in certain scenarios:

Data-Driven Approach: PINNs integrate observed or simulated data directly into the solution process, allowing for better handling of complex systems where analytical solutions or precise parameterization are challenging.

Flexibility: PINNs do not require domain discretization (e.g., creating a mesh), making them well-suited for problems with irregular geometries, high-dimensional spaces, or evolving boundaries.

Unified Framework: By incorporating both the governing PDEs and boundary/initial conditions as soft constraints in the loss function, PINNs can solve problems seamlessly without requiring separate formulations for different boundary conditions or regions.

Scalability: PINNs leverage the computational power of modern deep learning frameworks, allowing for efficient parallelization and scalability to larger problems or more complex domains.
Regularization and Generalization: PINNs naturally regularize solutions through their loss function, which enforces physics constraints. This reduces the risk of overfitting and ensures solutions remain physically meaningful even in the presence of noisy or incomplete data.

In our case, using PINNs for MFPT solutions enables a flexible and efficient method for predicting outcomes in transport equations, offering potential for faster computations, especially in high-dimensional or complex systems.

Using the Finite Element Method (FEM), we solved the MFPT partial differential equation (PDE). Below are visualizations of the FEM solution in a circular domain:
<div style="text-align: center;">
    <img src="Circledomain/Exist_from_both/Figs/FEMMFPT0.png" 
         alt="Circular Domain Example" 
         width="400" 
         style="display: inline-block; margin-right: 10px;" />
    <img src="Circledomain/Exitfromadisk/Figs/MFPTSOLcircle.png" 
         alt="Circular Domain Example" 
         width="400" 
         style="display: inline-block;" />
   </div>
   
The PINN model produced the following solutions for the same circular domain, demonstrating its ability to closely replicate the FEM results:

<div style="text-align: center;">
    <img src="Circledomain/Exist_from_both/Figs/Predicted_T_0.0000.png" 
         alt="Circular Domain Example" 
         width="500" 
         style="display: inline-block; margin-right: 10px;" />
    <img src="Circledomain/Exitfromadisk/Figs/PINNcircle.png" 
         alt="Circular Domain Example" 
         width="325" 
         style="display: inline-block;" />
   </div>

We compared the FEM and PINN solutions to the analytical solution by plotting their results along a specific cross-section of the domain. The comparison shows excellent agreement between the methods:
<div style="text-align: center;">
    <img src="Circledomain/Exist_from_both/Figs/Prediction_Exact_RightLineSolution_0.0000.png" 
         alt="Circular Domain Example" 
         width="400" 
         style="display: inline-block; margin-right: 10px;" />
    <img src="Circledomain/Exitfromadisk/Figs/PINNprediction.png" 
         alt="Circular Domain Example" 
         width="400" 
         style="display: inline-block;" />
   </div>


# Usage

This repository provides a comprehensive set of tools and scripts for reproducing the numerical simulations and examples detailed in the paper. Each section is accompanied by its corresponding implementation to ensure clarity and reproducibility.

Feel free to explore and modify the code to extend its applications or adapt it to other transport equation scenarios.
