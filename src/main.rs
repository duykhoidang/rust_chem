extern crate ndarray;
extern crate ndarray_linalg;

mod gaussint;
use gaussint::*;
use gaussians::Gaussian;

use rayon::prelude::*;
use ndarray::*;
use ndarray::prelude::*;
use ndarray_linalg::*;
use ndarray::parallel::*;
use rand::Rng;
use std::collections::vec_deque::VecDeque;

mod est;
use est::do_hartree_fock;

type M = Array2<f64>;

// mod gauss_int;

fn main() {
  // let n = 5;
  // let mut x = M::ones((n,n));
  // let mut rng = rand::thread_rng();
  // for i in x.iter_mut() {
  //   *i = rng.gen_range(-1.,1.);
  // }
  // let mut y = x.dot(&x.t());
  // for i in y.indexed_iter() {
  //   print!("{:8.4}",i.1);
  //   if i.0.1 == n-1 {
  //     println!();
  //   }
  // }
  // y = x.dot(&x.t());

  // println!("Testing eigh");
  // let a = arr2(&[[1.,0.5],[0.5,1.]]);
  //Note eigh returns column vectors
  // let (e,vecs) = y.eigh(UPLO::Lower).unwrap();
  // print!("Eigenvalues: ");
  // for i in e.iter() {
  //   print!(" {:8.4} ",i);
  // }
  // println!();
  // println!("Eigenvectors");
  // for i in vecs.indexed_iter() {
  //   print!("{:8.4}",i.1);
  //   if i.0.1 == n-1 {
  //     println!();
  //   }
  // }
  // println!("multiplication");
  // let z = y.dot(&vecs);
  // for i in z.indexed_iter() {
  //   print!("{:8.4}",i.1);
  //   if i.0.1 == n-1 {
  //       println!();
  //   }
  // }
  //checking obara-saika overlap
  // let z_a = 1.;
  // let x_a = [0.;3];
  // let i = [0i16;3];
  // let z_b = 1.;
  // let x_b = [0.;3];
  // let j = [0i16;3];

  // let sij = overlap(z_a, &x_a, &i, z_b, &x_b, &j);
  // let tij = ke_int(z_a, &x_a, &i, z_b, &x_b, &j);
  // let sij2 = overlap(z_a, &x_a, &[1i16;3],z_b,&x_b,&[1i16;3]);
  // let vij = nuc_e(z_a, &x_a, &i, z_b, &x_b, &j, &x_b);
  // let ijij = eri_prim(z_a, &x_a, &i, z_b, &x_b, &j,
  //    z_a, &x_a, &i, z_b, &x_b, &j);
  // println!("{} {} {} {} {}",sij,tij,sij2,vij,ijij);
  let g = Gaussian::new(
    1,
    [1.4,0.,0.],
    vec![1.],
    vec![1.],
    [0i16;3]
  );
  let h = Gaussian::new(
    1,
    [0.,0.,0.],
    vec![1.],
    vec![1.],
    [0i16;3]
  );

  // let sij = Gaussian::get_sij(&g,&g);
  // let sij2 = Gaussian::get_sij(&g,&h);
  // println!("{} {}",sij,sij2);

  // let mut q = M::zeros((n,n));

  // q.as_slice_mut()
  //   .par_iter_mut()
  //   .enumerate()
  //   .for_each(|(p,sij)| {
  //     sij[p] = Gaussian::get_sij(&g,&h);
  //   }
  // );

  // q.par_iter_mut().for_each(
  //   |s| *s = Gaussian::get_sij(&g,&h)
  // );
  
  // q.as_slice_mut().iter_mut().enumerate().for_each(
  //   |(r,j)| {
  //     j.par_iter_mut().enumerate().for_each(
  //       |(l,k)| {
  //         *k = 0f64 + l as f64;
  //       }
  //     );
  //   }
  // );
  // println!("{:?}",q);
  // let x = M::from_elem(q.dim(),2.);
  // println!("{:?}",q*x);
  // let x :f64 = q.par_iter().map(|i| i*i).into_par_iter().sum();
  // println!("{}",x);
  // let mut v = VecDeque::<M>::new();

  // v.push_back(M::ones((2,2)));
  // v.push_back(M::zeros((n,n)));
  // println!("{:?}",v);
  // v.pop_front();
  // println!("popped");
  // println!("{:?}",v);

  let mut coordn = Vec::<[f64;3]>::new();
  let mut z_nucs = Vec::<i16>::new();
  let mut ao_bas = Vec::<Gaussian>::new();
  let charge = 0i16;
  let nbasis = 2usize;
  let errmax = 1e-4;
  let n_diis = 4usize;
  let maxcyc = 100usize;

  // let var = Gaussian::get_ven(&g,&g,&vec![[1.4,0.,0.]],&vec![1 as i16]);
  // let var = gaussint::primint::nuc_e(1.,&[1.4,0.,0.],&[0i16;3],1.,&[1.4,0.,0.],&[0i16;3],&[1.4,0.,0.]);
  // println!("{}",var);

  ao_bas.push(g);
  ao_bas.push(h);
  coordn.push([1.4,0.,0.,]);
  coordn.push([0.,0.,0.]);
  z_nucs.push(1);
  z_nucs.push(1);

  do_hartree_fock(coordn,z_nucs,ao_bas,charge,2,1e-5,4,10,1e-5);
}
