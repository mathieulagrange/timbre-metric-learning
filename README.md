# Musical Timbre Perception Models: From Perceptual to Learned Approaches

**B. Pascal** and M. Lagrange

*Nantes Université, École Centrale Nantes, CNRS, LS2N, Nantes, France*

This self-contained toolbox accompanies the preprint: [hal-04501973](https://hal.science/hal-04501973v1/document)

1. The demonstration notebook [`learn-metric`](https://github.com/bpascal-fr/timbre-metric-learning/blob/main/learn-metric.ipynb) reproduces the explained variances reported in Table 1.
2. The robustness of the learning procedure can be tested using the [`robustness`](https://github.com/bpascal-fr/timbre-metric-learning/blob/main/robustness.ipynb) notebook.

Timbre, encompassing an intricate set of acoustic cues, is key to identify sound sources,  and especially to discriminate musical instruments and playing styles.
Psychoacoustic studies focusing on timbre deploy massive efforts to explain human timbre perception.
To uncover the acoustic substrates of timbre perceived dissimilarity, a recent work leveraged metric learning strategies on different perceptual representations and performed a meta-analysis of seventeen dissimilarity rated musical audio datasets.
By learning salient patterns in very high-dimensional representations, metric learning accounts for a reasonably large part of the variance in human ratings.
The present work shows that combining the most recent deep audio embeddings with a metric learning approach makes it possible to explains almost all the variance in human dissimilarity ratings.
Furthermore, the robustness of the learning procedure against simulated human rating variability is thoroughly investigated.
Intensive numerical experiments support the explanatory power and robustness against degraded training data of deep embeddings.


## Timbre audio datasets

Seventeen audio datasets are considered from eight historical studies focusing on human timbre perception, following [1]:

- Grey1977
- Grey1978
- Iverson1993_Whole
- Iverson1993_Onset
- Iverson1993_Remainder
- McAdams1995
- Lakatos2000_Harm
- Lakatos2000_Perc
- Lakatos2000_Comb
- Barthet2010
- Patil2012_A3
- Patil2012_DX4
- Patil2012_GD4
- Siedenburg2016_e2set1
- Siedenburg2016_e2set2
- Siedenburg2016_e2set3
- Siedenburg2016_e3.

Audio samples [sounds](https://github.com/EtienneTho/musical-timbre-studies/tree/master/ext/python/sounds) and human dissimilarity ratings [data](https://github.com/EtienneTho/musical-timbre-studies/tree/master/ext/python/sounds) are downloaded from [github.com/EtienneTho/musical-timbre-studies](https://github.com/EtienneTho/musical-timbre-studies/) and placed untouched in [timbre-metric-learning/data](https://github.com/bpascal-fr/timbre-metric-learning/tree/main/data) for user convenience.

## Models of human audio timbre perception

Eight different embeddings are provided.

### Time-frequency representations

*Computed on audio samples by the authors.*

- `STFT`: the spectrogram is obtained from a short-time Fourier transform;
- `scattering`: the scattering transform [2] is computed using the toolbox [kymat.io](https://www.kymat.io/).

### Perceptual representations

*Computed from the codes provided in [github.com/EtienneTho/musical-timbre-studies](https://github.com/EtienneTho/musical-timbre-studies) associated to [1].*

- `cochlea`: the auditory representation modeling cochlea processing is implemented from [3];
- `STMF`: the implementation of the spectrotemporal modulation frequency follows the detailed description of [4].

### Learned representations

*Computed on audio samples by the authors.*

- `EnCodec`: introduced in [5], computed from the toolbox [github.com/facebookresearch/encodec](https://github.com/facebookresearch/encodec);
- `CLAP`: introduced in [6], computed from the toolbox [github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP);
- `MERT`: introduced in [7], computed from the toolbox [github.com/yizhilll/MERT](https://github.com/yizhilll/MERT).

## References

[1] Thoret, E., Caramiaux, B., Depalle, P., & Mcadams, S. (2021). Learning metrics on spectrotemporal modulations reveals the perception of musical instrument timbre. *Nature Human Behaviour*, 5(3), 369-377.

[2] Andreux M., Angles T., Exarchakis G., Leonarduzzi R., Rochette G., Thiry L., Zarka J., Mallat S., Andén J., Belilovsky E., Bruna J., Lostanlen V., Chaudhary M., Hirn M. J., Oyallon E., Zhang S., Cella C., Eickenberg M. (2020). Kymatio: Scattering Transforms in Python. *Journal of Machine Learning Research*, 21(60):1−6.

[3] Chi, T., Ru, P., & Shamma, S. A. (2005). Multiresolution spectrotemporal analysis of complex sounds. *The Journal of the Acoustical Society of America*, 118(2), 887-906.

[4] Patil, K., Pressnitzer, D., Shamma, S., & Elhilali, M. (2012). Music in our ears: the biological bases of musical timbre perception. *PLoS Computational Biology*, 8(11), e1002759.  

[5] Défossez, A., Copet, J., Synnaeve, G., & Adi, Y. (2022). High fidelity neural audio compression. *Preprint arXiv:2210.13438*.

[6] Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. (2023, June). Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation. *Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing*, (pp. 1-5). IEEE.

[7] Yizhi, L. I., Yuan, R., Zhang, G., Ma, Y., Chen, X., Yin, H., ... & Fu, J. (2023, October). MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training. *Proceedings of The Twelfth International Conference on Learning Representations*.
