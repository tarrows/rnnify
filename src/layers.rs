extern crate ndarray;
use ndarray::linalg::Dot;
use ndarray::prelude::*;

pub trait Layer<Din, Dout: ndarray::Dimension, T> {
  fn forward(&self, x: Array<f32, Din>) -> Array<f32, Dout>;
  fn backwawrd(&self, x: Array<f32, Dout>) -> Array<f32, Din>;
  fn get_params(&self) -> T;
  fn set_params(&self, params: &T) -> ();
}

pub trait AffineParams<Din, Dout, Dw: ndarray::Dimension> {
  fn get_weight(&self) -> Array<f32, Dw>; // Is it able to derive D3 from D1, D2?
  fn get_bias(&self) -> Array<f32, Dout>;
}

pub struct Affine<Din, Dout, Dw: ndarray::Dimension, P: AffineParams<Din, Dout, Dw>> {
  weight: Array<f32, Dw>,
  bias: Array<f32, Dout>,
  dummy_in: Array<f32, Din>,
  dummy_params: P,
}

// impl<Din, Dout, Dw: ndarray::Dimension, P: AffineParams<Din, Dout, Dw>> Affine<Din, Dout, Dw, P> {
//   fn forward(&self, x: Array<f32, Din>) -> Array<f32, Dout> {
//     x.dot(self.weight) + self.bias
//   }
// }
