# Inverse Weight-Balancing for Deep Long-Tailed Learning(AAAI2024)
Pytoch version of Inverse Weight-Balancing for Deep Long-Tailed Learning. IWB is a adaptive classifier design method that outperforms previous approaches.[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/29055)

# Train
The first stage of training： `python first-stage-training.py`

You can get naive model by this instruction, the model that be stored in './exp./stage_1/' will be used in next stage.

The second stage of training： `python second-stage-training.py`

You can get IWB model by this instruction, the model will be store in './exp./stage_2/'.

# Result
There is an interesting phenomenon that the weight-norms distribution of the classifier is consistent with the logarithmic
distribution of the number of samples, as shown in this Figure.

# Requirements
This work only need 4*NVIDIA GeForce RTX 2080.
* Python version: 3.7.4
* PyTorch verion: 1.7.1

If you find our woke useful, please cite our work.

```
@inproceedings{dang2024inverse,
  title={Inverse Weight-Balancing for Deep Long-Tailed Learning},
  author={Dang, Wenqi and Yang, Zhou and Dong, Weisheng and Li, Xin and Shi, Guangming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={10},
  pages={11713--11721},
  year={2024}
}
```

# Acknowledgment
This work is heavily inspired by the following work: [CVPR2022](https://github.com/ShadeAlsha/LTR-weight-balancing)
