# Ceviche Challenges: Photonic Inverse Design Suite

The `ceviche_challenges` module contains a suite of photonic inverse design
challenge problems, backed by the open source
[`ceviche`](https://github.com/fancompute/ceviche/) simulator. These challenge
problems may be used to benchmark the ability of different topology optimization
algorithms to design relevant photonic components. The suite includes several
integrated photonic components, including a waveguide beam splitter, a waveguide
mode converter, a waveguide bend, and a wavelength division multiplexer (WDM).

The code in this package was used to produce the results in the
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

## Citation

```
@article{schubert_inverse_2022,
  title={Inverse design of photonic devices with strict foundry fabrication constraints},
  author={Schubert, Martin F and Cheung, Alfred KC and Williamson, Ian AD and Spyra, Aleksandra and Alexander, David H},
  journal={arXiv preprint arXiv:2201.12965},
  year={2022}
}
```
