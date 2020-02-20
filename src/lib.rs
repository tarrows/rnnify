extern crate ndarray;
use ndarray::prelude::*;

fn sigmoid_f32(x: f32) -> f32 {
  1.0 / f32::exp(-x)
}

pub fn sigmoid<D: ndarray::Dimension>(x: Array<f32, D>) -> Array<f32, D> {
  return x.mapv(sigmoid_f32);
}

#[test]
fn it_works() {
  let a = arr2(&[[1., 2.], [3., 4.]]);
  // let b = arr2(&[[2.7182817, 7.389056], [20.085537, 54.59815]]);
  assert!(a.iter().all(|x| x > &0.));
}
