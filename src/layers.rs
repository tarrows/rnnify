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

#[test]
fn test_sigmoid_it_works() {
  let a = array![1., 2.];
  let b = array![2.7182817, 7.389056];
  let sigm = Sigmoid {};
  assert!((sigm.forward(a) - b).iter().all(|x| x.abs() < 0.00001));
}
