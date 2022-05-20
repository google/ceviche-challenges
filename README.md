# Ceviche Challenges: Photonic Inverse Design Suite

The `ceviche_challenges` module contains a suite of photonic inverse design
challenge problems, backed by the open source
[`ceviche`](https://github.com/fancompute/ceviche/) finite difference frequency
domain (FDFD) simulator, and packaged into a standard API for calculating
scattering parameters and fields. Gradients and automatic differentiation
capabilities are provided by [HIPS autograd](https://github.com/HIPS/autograd).

The suite of challenge problems may be used to benchmark different topology 
optimization algorithms in designing relevant photonic components. The suite 
includes several integrated photonic components, including a waveguide beam 
splitter, a waveguide mode converter, a waveguide bend, and a wavelength
division multiplexer (WDM).

The code in this module was used to produce the results in the
[Inverse design of photonic devices with strict foundry fabrication constraints](https://arxiv.org/abs/2201.12965)
paper.

## Challenge problems

### Waveguide bend

![Waveguide bend](/img/waveguide_bend.png)

### Beam splitter

![Beam splitter](/img/beam_splitter.png)

### Mode converter

![Mode converter](/img/mode_converter.png)

### Wavelength division multiplexer (WDM)

![Wavelength division multiplexer (WDM)](/img/wavelength_divison_multiplexer.png)

## Installation

`ceviche_challenges` can be installed via `pip`:

```
pip install ceviche_challenges
```

## Usage

 `ceviche_challenges` provides several "prefab" component configurations, which
 live in the following submodules:

*   `ceviche_challenges.beam_splitter.prefabs`
*   `ceviche_challenges.mode_converter.prefabs`
*   `ceviche_challenges.waveguide_bend.prefabs`
*   `ceviche_challenges.wdm.prefabs`

Each component type (i.e. beam splitter, mode converter, etc) has an associated
model class for performing simulations and returning complex-valued scattering
parameters. An example using the 2 um x 2 um waveguide bend prefab is shown
below.

```python
import numpy as np
import ceviche_challenges

spec = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_2umx2um_spec()
params = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_sim_params()
model = ceviche_challenges.waveguide_bend.model.WaveguideBendModel(params, spec)

# The model class provides a convenience property, `design_variable_shape
# which specifies the design shape it expects.
design = np.ones(model.design_variable_shape)

# The model class has a `simulate()` method which takes the design variable as
# an input and returns scattering parameters and fields.
s_params, fields = model.simulate(design)
```

The model class can also be used inside of an objective function to perform an
optimization. Gradients and loss function automatic differentiation is supported
by [HIPS autograd](https://github.com/HIPS/autograd).

```python
import autograd
import autograd.numpy as npa

# Construct a loss function, assuming the `model` and `design` from the code
# snippet above are instantiated.

def loss_fn(x):
  """A simple loss function taking mean s11 - mean s21."""
  s_params, _ = model.simulate(x)
  s11 = npa.abs(s_params[:, 0, 0])
  s21 = npa.abs(s_params[:, 0, 1])
  return npa.mean(s11) - npa.mean(s21)

loss_value, loss_grad = autograd.value_and_grad(loss_fn)(design)
```

There are a number of other features in the device model class, such as
returning an image of the structure. These features are best explored by reading
the code and the function docstrings, as we don't have comprehensive standalone
documentation at this time.

## Citation

If you find `ceviche_challenges` to be useful for your research, please consider
 citing our [paper](https://arxiv.org/abs/2201.12965). A BibTex citation is
 provided below for convenience.

```
@article{schubert_inverse_2022,
  title={Inverse design of photonic devices with strict foundry fabrication constraints},
  author={Schubert, Martin F and Cheung, Alfred KC and Williamson, Ian AD and Spyra, Aleksandra and Alexander, David H},
  journal={arXiv preprint arXiv:2201.12965},
  year={2022}
}
```
