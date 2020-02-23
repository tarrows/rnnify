extern crate ndarray;
use ndarray::prelude::*;

pub trait Layer<Din, Dout: ndarray::Dimension, T> {
  fn forward(&self, x: Array<f32, Din>) -> Array<f32, Dout>;
  fn backwawrd(&self, x: Array<f32, Dout>) -> Array<f32, Din>;
  fn get_params(&self) -> T;
  fn set_params(&self, params: &T) -> ();
}

pub struct Affine {
  weight: Array2<f32>,
  bias: Array1<f32>,
}

impl Affine {
  pub fn forward(&self, x: Array1<f32>) -> Array1<f32> {
    x.dot(&self.weight) + &self.bias
  }
}

pub struct Sigmoid {}
impl Sigmoid {
  pub fn forward(&self, x: Array1<f32>) -> Array1<f32> {
    x.mapv(|x| 1.0 / f32::exp(-x))
  }
}

#[test]
fn test_affine_it_works() {
  let x_in = array![0., 1., 2.];
  let weight = array![[0., 1.], [2., 3.], [4., 5.],];
  let bias = array![0., 1.];
  let affine = Affine { weight, bias };
  assert_eq!(affine.forward(x_in), array![10., 14.,]);
}
