extern crate ndarray;
use ndarray::prelude::*;

fn sigmoid_f32(x: f32) -> f32 {
  1.0 / f32::exp(-x)
}

pub fn sigmoid<D: ndarray::Dimension>(x: Array<f32, D>) -> Array<f32, D> {
  return x.mapv(sigmoid_f32);
}

#[test]
fn it_works() {}
