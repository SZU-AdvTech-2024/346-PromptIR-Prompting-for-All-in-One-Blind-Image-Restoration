##### I. This reproduction project mainly refers to : [Github](https://github.com/va1shn9v/PromptIR)
Hint: The original PromptIR project provides a relatively bloated `env.yml` for configuring the environment, which prones to failure. We recommend the following approach instead:
1. Install appropriate `torch, torchvision` and `pytorch-lightning` in your environment, which are the main dependencies for running the project. 
For our environment (which requires CUDA>=11.1):
    ```
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    conda install lightning -c conda-forge
    ```
1. Try running the project and install any missing dependencies as prompted using `pip install` or `conda install`. In our practice, the following requirements need to be installed: `wandb, einops, matplotlib, scikit-image`.
This method can ensure to the greatest extent that the environment is correctly configured to suit your own device.


##### II. The DAVIS 2016 dataset can be downloaded from：[DAVIS 2016](https://davischallenge.org/davis2016/code.html)
To create compressed video frames using ffmpeg, please refer to `ffmpeg.txt`

