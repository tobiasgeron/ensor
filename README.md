# ensor

[![codecov](https://codecov.io/gh/tobiasgeron/ensor/graph/badge.svg)](https://codecov.io/gh/tobiasgeron/ensor)

Developed by Tobias Géron at the University of Toronto around Fall 2026. The goal of this code is to help with creating pretty colour images of astronomical observations. 

I've had to create colour images of galaxies a few too many times. Each time, I would dig up old code, try to decipher how it worked, spend too much time trying out different scalings, and figure out what exactly I needed to update. Last week, I realized this could probably be done more efficiently. I decided to create a generalizable system and put everything in a .py file, so that I can use this in the future and save myself some time. Hopefully this saves someone else some time, too.

To be clear; this is nothing revolutionary. I just wanted my own version of these colouring codes (see below) that I fully understood and can easily manipulate. In short, this is a lightweight and flexible code, which allows users to easily define their own weights and stretch/scaling functions, and help fine-tune these scaling parameters.

This code is heavily inspired by [Lupton+2004](https://arxiv.org/abs/astro-ph/0312483), the [Legacy Survey pipeline](https://github.com/legacysurvey/legacypipe/blob/f77ed93c4e9df40e27aec132fd9a9db0c78678d5/py/legacypipe/survey.py#L719), and Astropy's [make_lupon_rgb](https://docs.astropy.org/en/stable/api/astropy.visualization.make_lupton_rgb.html). 
    



### How to install

The simplest way to use this code is to just download the `ensor.py` file, copy-paste in your project and import it directly. However, this can be prone to dependency conflicts. 

Alternatively, I recommend [git cloning](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) this entire repository and navigating into it by running these commands in the terminal:

```
git clone https://github.com/tobiasgeron/ensor.git
cd ensor
```

and [creating a separate conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for ensor and activating it:

```
conda create --name venv_ensor python=3.14
conda activate venv_ensor
```

Then run this command to install the correct dependencies:

```
pip install -r requirements.txt
```

This code was last tested with python 3.14. The main dependencies are numpy, matplotlib, and astropy.



### Minimal example

Below you can find a minimal example of how to use this code. However, it is strongly encouraged to have a look at the more in-depth tutorials as well.

First, import the package.

```
from ensor import *
```

Then, assuming you've loaded g, r, and i images (in nanomaggies) into the corresponding variables:


```
imgs = [g,r,z]
bands = ['g','r','z']

weights = dict( # map bands into corresponding r,g,b channels. 
    g = (0, 0, 6.0), 
    r = (0, 3.4, 0),
    z = (2.2, 0, 0),
)

rgb = make_image(imgs, bands, weights, func = asinh) # the main functionofs this code
plot_image(rgb)
```

Please do check out the tutorials!



### Why "ensor"?

This code was named after [James Ensor](https://en.wikipedia.org/wiki/James_Ensor), a famous Belgian painter. I imagine he would have less trouble with colouring galaxies than me. 