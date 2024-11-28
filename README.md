# Mean-First-Passage-Times-for-Transport-Equations
Code for the Mean First Passage Time Paper 

This repository contains the code used in the research presented in the paper \textit{Mean First Passage Times for Transport Equations}. For more details on the paper, please refer to the following link: https://arxiv.org/abs/2404.00400

The code in this repository includes the implementation for the example involving a circular domain, as described in the paper. Specifically, it addresses the solution for particles traveling along microtubules. It also includes the implementation for the example involving a square domain in addresses the solution for wolves moving in the wild

Using the Finite Difference Method (FDM), we verify our numerical simulations with high accuracy. Below is the Mean First Passage Time (MFPT) solution for the circular domain example:
![Alt Text](Circledomain/Exitfromadisk/Figs/MFPTSOLcircle.tiff)

Furthermore, we demonstrate that Physics-Informed Neural Networks (PINNs) successfully recapture the solution for the "Exit from the Disk" section of the paper:
![Alt Text](Circledomain/Exitfromadisk/Figs/PINNcircle.png)
The full domain predictions using PINNs are shown below:
![Alt Text](Circledomain/Exitfromadisk/Figs/PINNprediction.png)
