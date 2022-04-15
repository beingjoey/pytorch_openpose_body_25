# pytorch_openpose_body_25
pytorch implementation of openpose including Body coco and body_25 Estimation, and the pytorch model is directly converted from openpose caffemodel by caffemodel2pytorch.I did some work to implement the body_25 net model and to figure out the correspond of Part Confidence Map and Part Affinity Field outputs. Some code came from PyTorch OpenPose, and I debug some problem.

Download the torch model ,and put them in the model folder(mkdir by yourself)

# Demo:

python3 demo.py images/timg.jpeg

# Downloads:
* [body_25](https://pan.baidu.com/s/1CopeW-Em4Tm9H-Wl_hzVfg) download code : 9g4p
* google cloud:(https://drive.google.com/file/d/1ghXakEXhBMCdV78K6tCFTPp_vjJDWmcE/view?usp=sharing)
* [body_coco](https://pan.baidu.com/s/19Hjo5qEsNPoRt6zY6Ly4Lw) download code : kav3
* google cloud:(https://drive.google.com/file/d/1VPiIxXk5KWEwdJlVVe5PDQ1QufMS1Zpk/view?usp=sharing)


## References
* [OpenPose doc](https://arxiv.org/abs/1812.08008)
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [PyTorch OpenPose](https://github.com/Hzzone/pytorch-openpose)

## License
* [OpenPose License](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
