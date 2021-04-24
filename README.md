# Introduction
This project aims to denoise retinal optical coherenece tomography (OCT) images , using a lightweight mimic autoencoder [1].
The method is detailed in ["A Lightweight Mimic Convolutional Auto-encoder for Denoising Retinal Optical Coherence Tomography Images"](https://ieeexplore.ieee.org/document/9399639)
# Dependencies
- Python 3.7+
- Tensorflow 1.15.2
- Keras
# Training dataset
There are Sample training data in Data folder. Noisy images have been acquired from [more data](https://misp.mui.ac.ir/fa/oct-topcon). Corresponding clean images have been produced using two states of the arts OCT image denoising methods. The first denoising method was proposed by Kafieh et al. [2] and is called 3DCWT-KSVD. The second denoising method was proposed by Amini and Rabbani [3] and is called GT-SC-GMM. You can find the source codes of 3DCWT-KSVD [here](https://sites.google.com/site/rahelekafieh/research/state-of-the-art-method-for-oct-denoising/code-tmi-oct-denoiing/CODE%20OCT%20DENOISING%20KAFIEH.rar?attredirects=0&d=1) and GT-SC-GMM [here] () and use them to produce more clean images.
# Test dataset
We used test images from various imaging devices (refer to Table I of refrenced paper[1]) to evaluate the proposed denoising method.  
- More test images acquired from Heidelberg imaging system can be found for [near fovea here](https://hrabbani.site123.me/available-datasets/dataset-for-oct-classification-50-normal-48-amd-50-dme) and for [near ONH here](https://hrabbani.site123.me/available-datasets/onh-based-oct-of-7-healthy-and-7-glaucoma-data-captured-by-heidelberg-spectralis).
- More test images acquired from TOPCON imaging system can be found for [near fovea here](https://misp.mui.ac.ir/fa/oct-topcon) and for [near ONH here](https://hrabbani.site123.me/available-datasets/onh-based-oct-of-7-healthy-and-7-glaucoma-data-captured-by-heidelberg-spectralis).
- More test images acquired from custom-made Basel imaging system can be found for [near fovea here](https://misp.mui.ac.ir/bank).

# Citation
If you find this work useful for your research, please cite our paper:
- <a id="1">[1]</a>
 M. Tajmirriahi, R. Kafieh, Z. Amini and H. Rabbani, "A Lightweight Mimic Convolutional Auto-encoder for Denoising Retinal Optical Coherence Tomography Images," in IEEE Transactions on Instrumentation and Measurement, doi: 10.1109/TIM.2021.3072109.
- <a id="2">[2]</a>
	R. Kafieh, H. Rabbani, I. Selesnick, Three dimensional data-driven multi scale atomic representation of optical coherence tomography, IEEE Trans. Med. Imaging. 34 (2014) 1042–1062.
- <a id="3">[3]</a>
Z. Amini, H. Rabbani, Optical coherence tomography image denoising using Gaussianization transform, J. Biomed. Opt. 22 (2017) 86011.
