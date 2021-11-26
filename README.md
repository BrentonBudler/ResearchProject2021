# ResearchProject2021
Detecting network intrusions is an imperative part of the modern cybersecurity landscape. Over the years, researchers have leveraged the ability of Machine Learning to identify and prevent network attacks. Recently there has been an increased interest in the applicability of Deep Learning in the network intrusion detection domain. However, Network Intrusion Detection Systems developed using Deep Learning approaches are being evaluated using the outdated KDD Cup '99 and NSL-KDD datasets which are not representative of real-world network traffic. Recent comparisons of these approaches on the more modern CSE-CIC-IDS2018 dataset, fail to address the the severe class imbalance in the dataset which leads to significantly biased results. By addressing this class imbalance and performing an experimental evaluation of a Deep Neural Network, Convolutional Neural Network and Long Short-Term Memory Network on the balanced dataset, this research provides deeper insights into the performance of these models in classifying modern network traffic data. The Deep Neural Network demonstrated the best classification performance with the highest accuracy (84.312\%) and F1-Score (83.799\%) as well as the lowest False Alarm Rate (2.615\%) whilst the Convolutional Neural Network was the most efficient model boasting the lowest model training and inference times. Dimensionality reduction using a Sparse AutoEncoder was incorporated to explore the potential benefits of the Self-Taught learning approach but did not improve model performance
[acc_f1_cic.pdf](https://github.com/BrentonBudler/ResearchProject2021/files/7607363/acc_f1_cic.pdf)

[ae_acc_f11_cic.pdf](https://github.com/BrentonBudler/ResearchProject2021/files/7607361/ae_acc_f11_cic.pdf)
