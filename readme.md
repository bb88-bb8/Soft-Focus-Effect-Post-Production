# Novel Computational Photography for Soft-Focus Effect in Automatic Post Production

## Introduction
This is a Python implementation of Algorithm 4 in my proposed paper, ``Novel Computational Photography for Soft-Focus Effect in Automatic Post Production, `` with my pre-generated kernel estimate matrix. Additionally, the low-resolution preview feature is included.
The program is with GUI and multi-thread programming for easy execution.

The code is tested under `Python 3.7.17` and ``Ubuntu 20.04`` and only currently supports three-channel images. Authentic B&W and four-channel images are not supported (only one channel).


## Install
To use the program please install the required package firt.

### To install PIL
``sudo apt-get install python-imaging-tk``

### To install tkinter
``sudo apt-get install python3-tk ``

### To install resting package 
``pip install -r requirement.txt``

## Instruction
To run the program, run ``python3 main.py``, then you will see the GUI, and you can choose any RGB image you want. If there is any error right after you select the image, please ensure that the image you chose is a three-channel image, not a four-channel (some PNG images might contain a transparent channel) or one-channel (B&W image).

## Demo
Click Select File to choose the desired image to generate a soft-focus effect.

Change the value of `M` to tune the major strength of the effect by dragging the bar. The `M` equals $\chi \times 10000$, where $\chi$ is defined in Eq. (32) in the paper.

Then, dragging the bar, choose the desired $(k,t)$ value in each channel. The higher $t$ can get a smoother effect. The higher $k$ can have a more substantial impact.

After the adjustment, click `Save Result` to store the full-resolution result.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./image/demo_1.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Demonstration  of the GUI environment</div>
</center>


## Citation

If you would like to cite this work, please use the following bib.
```
@ARTICLE{TSAISOFTFOCUS,
author={Tsai, Hao-Yu and Tsai, Morris C.-H. and Huang, Scott C.-H. and Wu, Hsiao-Chun},
journal={IEEE Transactions on Image Processing},
title={Novel Computational Photography for Soft-Focus Effect in Automatic Post Production},
year={2025},
volume={Early Access}
}
```
Moreover, if you have any questions about this project, please email to `s106021226@m106.nthu.edu.tw`.

## License
This code is released under the MIT License. 