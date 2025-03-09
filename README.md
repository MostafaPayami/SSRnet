The Python and MATLAB codes have been used for the experiments and results presented in the following paper [1]. 

[1] M. Payami, and S. D. Blostein, “Sparse Signal Recovery Neural Network with Application to High-Mobility Massive MIMO-OTFS Communication Systems” IEEE Trans. Veh. Tech., 2025. (Accepted Paper)

The Python codes are TensorFlow/Keras models for PositionNet+ and SSRnet. These codes are written by the authors of [1]. 

The MATLAB codes used for data generation of MIMO-OTFS systems, have been written by the authors of the following paper [2] downloaded from the following website: 
https://oa.ee.tsinghua.edu.cn/dailinglong/. 
If you use this MATLAB code package, please cite [2]:

[2] W. Shen, L. Dai, J. An, P. Fan, and R. W. Heath, “Channel estimation for orthogonal time frequency space (OTFS) massive MIMO”, IEEE Trans. Signal Process., 2019.

The following software and libraries are required:
1) TensorFlow: Version 2.18 or later
2) Python: Version 3.12 or later
3) MATLAB: Version R2014b or later

***************************************************************************************************

Paper's Abstract: 

A deep learning-based sparse signal recovery network SSRnet is designed. This network is built on the proposed neural network PositionNet+, which takes the received signal as input and obtains the support of the desired sparse matrix without requiring a sensing matrix. Using PositionNet+, SSRnet is able to recover the sparse signal precisely, outperforming conventional methods including perfect least-squares (LS) estimation by virtue of its denoising behavior, while offering substantially reduced computation. The network is then utilized to perform the channel estimation of high-mobility massive multiple-input multiple-output orthogonal time frequency space (MIMO-OTFS) wireless systems which is cast as a sparse signal recovery problem. In OTFS, data is modulated in the delay-Doppler domain to transform a fast time-varying and frequency-selective fading channel into a quasi-static and sparse channel. To achieve the optimal performance, OTFS systems require accurate channel estimation and low pilot signaling which are provided by SSRnet. Simulation and computational comparisons demonstrate that the proposed approach provides improved performance in terms of bit error rate (BER) and normalized mean squared error (NMSE), a reduction in pilot symbols overhead, as well as reduced computation. 

***************************************************************************************************
