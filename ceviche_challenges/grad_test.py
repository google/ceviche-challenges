import numpy as np
import ceviche_challenges

import autograd
import autograd.numpy as npa

import jax
import jax.numpy as jnp

from absl.testing import absltest

class JaxGradientsTest(absltest.TestCase):
    def _test_jax_model(self, spec, params, model, jax_model):
        design = np.ones(model.design_variable_shape)

        def loss_fn(x):
            """A simple loss function taking mean s11 - mean s21."""
            s_params, _ = model.simulate(x)
            s11 = npa.abs(s_params[:, 0, 0])
            s21 = npa.abs(s_params[:, 0, 1])
            return npa.mean(s11) - npa.mean(s21)

        def loss_fn_jax(x):
            """A simple loss function taking mean s11 - mean s21."""
            s_params, _ = jax_model.simulate(x)
            s11 = jnp.abs(s_params[:, 0, 0])
            s21 = jnp.abs(s_params[:, 0, 1])
            return jnp.mean(s11) - jnp.mean(s21)
    
        loss_value, loss_grad = autograd.value_and_grad(loss_fn)(design)
        jax_loss_value, jax_loss_grad = jax.value_and_grad(loss_fn_jax)(design)

        self.assertTrue(np.allclose(jax_loss_value, loss_value))
        self.assertTrue(np.allclose(jax_loss_grad, loss_grad))

    def test_grad_wg_bend(self):
        """Tests Waveguide Bend Model"""
        spec = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_2umx2um_spec()
        params = ceviche_challenges.waveguide_bend.prefabs.waveguide_bend_sim_params()
        model = ceviche_challenges.waveguide_bend.model.WaveguideBendModel(params, spec)
        jax_model = ceviche_challenges.waveguide_bend.jax_model.WaveguideBendModel(params, spec)

        self._test_jax_model(spec, params, model, jax_model)


    def test_grad_wdm(self):
        """Tests WDM Model"""
        spec = ceviche_challenges.wdm.prefabs.wdm_spec()
        params = ceviche_challenges.wdm.prefabs.wdm_sim_params()
        model = ceviche_challenges.wdm.model.WdmModel(params, spec)
        jax_model = ceviche_challenges.wdm.jax_model.WdmModel(params, spec)

        self._test_jax_model(spec, params, model, jax_model)

    def test_grad_mode_converter(self):
        """Tests Mode Converter Model"""
        spec = ceviche_challenges.mode_converter.prefabs.mode_converter_spec_12()
        params = ceviche_challenges.mode_converter.prefabs.mode_converter_sim_params()
        model = ceviche_challenges.mode_converter.model.ModeConverterModel(params, spec)
        jax_model = ceviche_challenges.mode_converter.jax_model.ModeConverterModel(params, spec)

        self._test_jax_model(spec, params, model, jax_model)

    def test_grad_beam_splitter(self):
        """Tests Beam Splitter Model"""
        spec = ceviche_challenges.beam_splitter.prefabs.pico_splitter_spec()
        params = ceviche_challenges.beam_splitter.prefabs.pico_splitter_sim_params()
        model = ceviche_challenges.beam_splitter.model.BeamSplitterModel(params, spec)
        jax_model = ceviche_challenges.beam_splitter.jax_model.BeamSplitterModel(params, spec)

        self._test_jax_model(spec, params, model, jax_model)

if __name__ == '__main__':
  absltest.main()
