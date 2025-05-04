
# Chosen Topic
Deep Learning-Based Encrypted Traffic Classification and Prediction

# Abstract
I plan to investigate the application of deep learning algorithms for the classification and prediction of encrypted traffic, focusing on encrypted traffic data under a specific communication protocol (e.g., UDP or TCP). Encrypted traffic, due to the complexity of its underlying protocols and the lack of visible payload information, presents significant challenges for traditional classification methods. By leveraging statistical features and time-series patterns, I aim to design a simple framework for classifying and forecasting encrypted traffic.

This work is inspired by three key papers. The first, *Robust Network Traffic Classification*, illustrates the limitations of classical pattern recognition techniques when handling encrypted and zero-day traffic, thereby highlighting the need for more adaptive approaches. The second, *Deep Learning for Encrypted Traffic Classification: An Overview*, emphasizes the unique characteristics of encrypted traffic and demonstrates how conventional neural network architectures, such as CNNs and RNNs, can effectively extract hidden spatial and temporal features, particularly in HTTPS traffic. The third paper, *Deep-Full-Range: A Deep Learning Based Network Encrypted Traffic Classification and Intrusion Detection Framework*, innovatively integrates multiple deep learning models (1D CNN, LSTM, SAE) into a unified framework, achieving significant improvements in both classification and intrusion detection.

In my final project, I plan to use public datasets to analyze encrypted traffic from a specific protocol (maybe UDP). The project will begin with the implementation of a CNN-based model for feature extraction and classification, and later, I intend to develop a temporal analysis module, comparing its performance with a reproduced traditional pattern recognition algorithm.

# References
[1] J. Zhang, X. Chen, Y. Xiang, W. Zhou, and J. Wu, “Robust Network Traffic Classification,” _IEEE/ACM Trans. Networking_, vol. 23, no. 4, pp. 1257–1270, Aug. 2015, doi: [10.1109/TNET.2014.2320577](https://doi.org/10.1109/TNET.2014.2320577).

[2] S. Rezaei and X. Liu, “Deep Learning for Encrypted Traffic Classification: An Overview,” _IEEE Commun. Mag._, vol. 57, no. 5, pp. 76–81, May 2019, doi: [10.1109/MCOM.2019.1800819](https://doi.org/10.1109/MCOM.2019.1800819).

[3] Y. Zeng, H. Gu, W. Wei, and Y. Guo, “Deep-Full-Range: A Deep Learning Based Network Encrypted Traffic Classification and Intrusion Detection Framework,” _IEEE Access_, vol. 7, pp. 45182–45190, 2019, doi: [10.1109/ACCESS.2019.2908225](https://doi.org/10.1109/ACCESS.2019.2908225).