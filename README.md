# TeLU Activation Function: A Drop-in Replacement for ReLU

We are excited to introduce **TeLU**, a new **universal and computationally efficient activation function** designed to serve as a drop-in replacement for ReLU. TeLU offers a compelling combination of theoretical and empirical benefits that make it a strong candidate for modern deep learning tasks.

## Few Key Benefits of TeLU

- **Better Generalization**: Achieves superior performance on unseen data compared to ReLU, ensuring models learn robust patterns.
- **Enhanced Stability**: Improves numerical stability during training, particularly for deep architectures.
- **Robustness**: Demonstrates resistance to perturbations and adversarial attacks, ensuring reliable model predictions.
- **Faster Convergence**: Accelerates training time, reducing computational costs without sacrificing accuracy.

## Why Choose TeLU?

TeLU closely mirrors the functional space of ReLU, making it **more similar to ReLU than other existing activation functions** such as GELU, ELU, SWISH, or MISH. This ensures that networks optimized for ReLU can easily adopt TeLU with **minimal tuning**, retaining or even improving their performance.

## Compatibility

TeLU is designed to work seamlessly with networks that were originally optimized for ReLU, making it an ideal drop-in replacement. Its simplicity and efficiency make it suitable for a wide range of tasks and architectures.

## How to Use TeLU

To use TeLU in your neural networks, simply replace your ReLU activation layers with TeLU. No additional modifications are needed, allowing for easy integration into your existing codebase.

## Citation

If you use TeLU or our implementation in your work, please cite the following paper:

```bibtex
@misc{fernandez2025teluactivationfunctionfast,
      title={TeLU Activation Function for Fast and Stable Deep Learning}, 
      author={Alfredo Fernandez and Ankur Mali},
      year={2025},
      eprint={2412.20269},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.20269}, 
}
