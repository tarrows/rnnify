extern crate ndarray;
// extern crate ndarray_rand;
use ndarray::prelude::*;
// use ndarray_rand::rand_distr::Normal;
// use ndarray_rand::RandomExt;

// pub trait Layer<Din, Dout: ndarray::Dimension, T> {
//   fn forward(&self, x: Array<f32, Din>) -> Array<f32, Dout>;
//   fn backwawrd(&self, x: Array<f32, Dout>) -> Array<f32, Din>;
//   fn get_params(&self) -> T;
//   fn set_params(&self, params: &T) -> ();
// }

pub trait Layer {
  fn forward(&self, x: Array1<f32>) -> Array1<f32>;
}

pub struct Affine {
  weight: Array2<f32>,
  bias: Array1<f32>,
}

impl Layer for Affine {
  fn forward(&self, x: Array1<f32>) -> Array1<f32> {
    x.dot(&self.weight) + &self.bias
  }
}

pub struct Sigmoid {}

impl Layer for Sigmoid {
  fn forward(&self, x: Array1<f32>) -> Array1<f32> {
    x.mapv(|x| 1.0 / f32::exp(-x))
  }
}

pub struct TwoLayerNet {
  w1: Array2<f32>,
  b1: Array1<f32>,
  w2: Array2<f32>,
  b2: Array1<f32>,
}

impl TwoLayerNet {
  // pub fn new(&self, input_size: u32, hidden_size: u32, output_size: u16) -> TwoLayerNet {
  //   // let i = input_size;
  //   // let h = hidden_size;
  //   // let o = output_size;
  //   // if not (2, 3) but (i, h) ...
  //   // the trait bound `(u32, u32): ndarray::Dimension` is not satisfied
  //   let w1 = Array2::random((2, 3), Normal::new(0., 1.).unwrap());
  //   let b1 = Array1::random(3, Normal::new(0., 1.).unwrap());
  //   let w2 = Array2::random((3, 2), Normal::new(0., 1.).unwrap());
  //   let b2 = Array1::random(2, Normal::new(0., 1.).unwrap());

  //   TwoLayerNet { w1, b1, w2, b2 }
  // }
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
