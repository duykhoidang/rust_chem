use linalg::general_mat_mul;
use ndarray::*;
use ndarray_linalg::*;
use ndarray::parallel::prelude::*;
// use ndarray::Zip;
// use rayon::prelude::*;
use std::collections::vec_deque::VecDeque;

use super::gaussint;
use gaussint::*;
use gaussians::Gaussian;

type M = Array2<f64>;

// pub struct EST {
//   coordn : Vec<[f64;3]>,//vector of nuc coordinates
//   zns : Vec<i16>,       //vector of nuc charges
//   aos : Vec<Gaussian>,  //vector of ao basis
//   nelec : u32,          //number of electrons
//   charge : i16,         //charge
//   energy : f64,         //energy from latest cycle
//   old_en : f64,         //energy from previous cycle
//   nbas : usize,           //number of basis functions
//   smat : M,   //overlap matrix S
//   hcore : M,  //core hamiltonian
//   cmat : M,   //AO coefficients
//   pmat : M,   //density matrix
//   oldp : M,   //previous density
//   gmat : M,   //G for HF method
//   fmat : M,   //Fock matrix
//   xmat : M,   //Matrix for orthogonalization
//   errmax : f64,         //max error in energy/density
//   n_diis : u8,          //number of density matrices in diis
//   diis_p : Vec<M>,  //stored diis density matrices
//   nuc_e : f64,          //nuclear energy
// }

pub fn do_hartree_fock(
  coordn : Vec<[f64;3]>,
  z_nucs : Vec<i16>,
  ao_bas : Vec<Gaussian>,
  charge : i16,
  nbasis : usize,
  errmax : f64,
  n_diis : usize,
  maxcyc : usize,
  eermax : f64
) { //need to implement oldp stuff
  let mut smat = M::zeros((nbasis,nbasis));
  let mut hcore = M::zeros((nbasis,nbasis));
  let mut cmat = M::zeros((nbasis,nbasis));
  let mut pmat = M::zeros((nbasis,nbasis));
  // let mut oldp = M::zeros((nbasis,nbasis));
  let mut gmat = M::zeros((nbasis,nbasis));
  let mut fmat = M::zeros((nbasis,nbasis));
  let mut xmat = M::zeros((nbasis,nbasis));
  let mut ftrans = M::zeros((nbasis,nbasis));
  let mut tmp1 = M::zeros((nbasis,nbasis));
  let mut tmp2 = M::zeros((nbasis,nbasis));
  let mut tmp3 = M::zeros((nbasis,nbasis));
  // let mut tmp4 = M::zeros((nbasis,nbasis));
  let ndiis = std::cmp::min(10 as usize,n_diis);
  let mut ferr : VecDeque<M> = VecDeque::with_capacity(ndiis);
  let mut err : VecDeque<M> = VecDeque::with_capacity(ndiis);
  ferr.push_front(M::zeros((nbasis,nbasis)));
  err.push_front(M::zeros((nbasis,nbasis)));
  let mut diis_count = 0usize;
  let mut nelec = 0i16;
  for z in z_nucs.iter() {
    nelec += z;
  }
  nelec -= charge;

  let mut old_energy;
  let mut energy = 0f64;

  get_smat(&mut smat,&ao_bas);
  get_hcore(&mut hcore,&ao_bas,&z_nucs,&coordn);
  get_xmat(&smat,&mut xmat,&mut tmp1);
  get_fmat(&mut fmat, &hcore, &gmat);

  let mut dens_conv = false;
  let mut e_conv = false;
  let mut diis_conv = false;
  let mut iter = 0usize;

  while !dens_conv && !e_conv && iter < maxcyc && !diis_conv {
    old_energy = energy;
    // oldp = pmat.clone();
    diis_conv = diis_iter(&mut fmat,&mut ferr,&pmat,&smat,&xmat,
      &mut tmp1, &mut tmp2,&mut tmp3,&mut err,diis_count,errmax
    );

    get_cmat(&mut ftrans,&mut cmat, &fmat, &xmat, &mut tmp1);
    get_pmat(&mut pmat,&cmat,nelec);
    get_gmat(&mut gmat,&pmat,&ao_bas);
    get_fmat(&mut fmat,&hcore,&gmat);

    energy = get_energy(&fmat,&hcore,&pmat);
    
    if (energy-old_energy).abs() < eermax {
      e_conv = true;
    }
    dens_conv = false;
    println!("{} {}",iter,energy);
    println!("{:?}",cmat);
    println!("{:?}",fmat);
    iter += 1;
    diis_count += 1;
    diis_count = diis_count%ndiis;
  }
}

fn get_smat(
  smat : &mut M,
  basis : &Vec<Gaussian>
) {
  let n = basis.len();
  // smat.iter_mut().enumerate().for_each(
  //   |(p,sij)| {
  //     let i = p/n;
  //     let j = p%n;
  //     *sij = Gaussian::get_sij(&basis[i],&basis[j]);
  //   }
  // );
  //stupid trick until par_iter_mut().enumerate() 
  //supported for ndarray

  smat.as_slice_mut().iter_mut().for_each(
    |mat| {
      mat.par_iter_mut().enumerate().for_each(
        |(p,sij)| {
          let i = p/n;
          let j = p%n;
          *sij = Gaussian::get_sij(&basis[i],&basis[j]);
        }
      );
    }
  )

}

fn get_hcore(
  hcore : &mut M,
  basis : &Vec<Gaussian>,
  z_nucs : &Vec<i16>,
  coordn : &Vec<[f64;3]>
) {
  let n = basis.len();
  hcore.as_slice_mut().iter_mut().for_each(
    |mat| {
      mat.par_iter_mut().enumerate().for_each(
        |(p,hij)| {
        let i = p/n;
        let j = p%n;
        *hij = Gaussian::get_tij(&basis[i],&basis[j]);
        *hij += Gaussian::get_ven(&basis[i],&basis[j], &coordn, &z_nucs);
        }
      )
    }
  );
}

fn get_xmat(
  smat : &M,
  mut xmat : &mut M,
  mut tmp : &mut M
) {
  let (e,umat) = smat.eigh(UPLO::Lower).unwrap();
  let n = smat.shape()[0];
  let mut sdiag = M::eye(n);
  for (i,ev) in sdiag.iter_mut().step_by(n+1).zip(e.iter()) {
    *i = (1f64 / *ev).sqrt();
  }
  general_mat_mul(1f64, &umat,&sdiag,0f64,&mut tmp);
  general_mat_mul(1f64,&tmp,&(umat.t()),0f64,&mut xmat);
}

fn get_fmat(
  fmat : &mut M,
  hcore : &M,
  gmat : &M
) {
  *fmat = hcore + gmat;
}

fn get_cmat(
  mut ftrans : &mut M,
  mut cmat : &mut M,
  fmat : &M,
  xmat : &M,
  mut tmp : &mut M
) {
  // let n = fmat.shape()[0];
  general_mat_mul(1f64,&xmat.t(),&fmat,0f64,&mut tmp);
  general_mat_mul(1f64,&tmp,&xmat,0f64,&mut ftrans);
  let (_e,ctrans) = ftrans.eigh(UPLO::Upper).unwrap();
  general_mat_mul(1f64,&xmat,&ctrans,0f64,&mut cmat);
}

fn get_pmat(
  pmat : &mut M,
  cmat : &M,
  nelec : i16,
) {
  let n = pmat.shape()[0];
  let ne = nelec as usize;
  pmat.as_slice_mut().iter_mut().for_each(
    |pm| {
      pm.par_iter_mut().enumerate().for_each(
        |(p,pij)| {
          let i = p/n;
          let j = p/n;
          let rowi = cmat.row(i);
          let rowj = cmat.row(j);
          let mut sum = 0f64;
          for (_a,(ia,ja)) in (0..(ne/2)).zip(rowi.iter().zip(rowj.iter())) {
            sum += ia * ja;
          }
          *pij = sum * 2f64;
        }
      )
    }
  );
}

fn get_gmat(
  gmat : &mut M,
  pmat : &M,
  basis : &Vec<Gaussian>
) {
  //making slow code first
  let n = gmat.shape()[0];
  for i in 0..n {
    for j in 0..n {
      let mut total = 0f64;
      for k in 0..n {
        for l in 0..n {
          let mut tmp = 0f64;
          tmp += Gaussian::get_eri(&basis[i],&basis[j],&basis[k],&basis[l]);
          tmp -= 0.5*Gaussian::get_eri(&basis[i],&basis[k],&basis[l],&basis[j]);
          tmp *= pmat[[k,l]];
          total += tmp;
        }
      }
      gmat[[i,j]] = total;
    }
  }
}

fn get_energy(
  fmat : &M,
  hcore : &M,
  pmat : &M
) -> f64 {
  let mut energy = 0f64;
  let n = fmat.shape()[0];
  for i in 0..n {
    for j in 0..n {
      let mut tmp = hcore[[i,j]] + fmat[[i,j]];
      tmp *= pmat[[j,i]];
      energy += 0.5*tmp;
    }
  }
  return energy;
}

fn diis_iter(
  fmat : &mut M,
  ferr : &mut VecDeque<M>,
  pmat : &M,
  smat : &M,
  xmat : &M,
  mut fps : &mut M,
  mut spf : &mut M,
  mut tmp : &mut M,
  err : &mut VecDeque<M>,
  diis_count : usize,
  errmax : f64
) -> bool {
  let mut dens_conv = false;
  if diis_count + 1 > ferr.len() {
    ferr.push_back(fmat.clone());
  }
  if diis_count + 1 > err.len() {
    err.push_back(M::zeros(fmat.dim()));
  }

  //fps = F@P@S
  general_mat_mul(1f64,&fmat,&pmat,0f64,&mut tmp);
  general_mat_mul(1f64,&tmp,&smat,0f64,&mut fps);
  //spf = S@P@F
  general_mat_mul(1f64,&smat,&pmat,0f64,&mut tmp);
  general_mat_mul(1f64,&tmp,&fmat,1f64,&mut spf);
  //diis err = X.T@(fps-spf)@X
  subtract(&mut err[diis_count],&fps,&spf);
  general_mat_mul(1f64,&xmat.t(),& err[diis_count],0f64,&mut tmp);
  general_mat_mul(1f64,&tmp,&xmat,0f64,&mut err[diis_count]);

  let err_norm : f64 = err[diis_count].par_iter()
    .map(|i| i*i).into_par_iter().sum();
  
  if err_norm < errmax {
    dens_conv = true;
    return dens_conv;
  }
  let n = err.len();
  let mut lmat = M::from_elem((n+1,n+1),-1.);
  lmat[[n,n]] = 0f64;
  for i in 0..n {
    for j in 0..n {
      lmat[[i,j]] = (&err[i]*&err[j]).par_iter()
        .map(|val| val * val).into_par_iter().sum();
    }
  }

  let mut tmp_v = Array1::<f64>::zeros(n+1);
  tmp_v[n] = -1f64;
  fmat.fill(0f64);
  let x = lmat.solve_into(tmp_v).unwrap();
  for (fock,coeff) in ferr.iter().zip(x.iter()) {
    fmat.as_slice_mut().iter_mut().zip(fock.as_slice().iter()).for_each(
      |(nfock,ofock)| {
        nfock.par_iter_mut().zip(ofock.par_iter()).for_each(
          |(n,o)| *n += coeff * o
        )
      }
    )
  }

  return dens_conv;
}

fn subtract(
  diff : &mut M,
  lmat : &M,
  rmat : &M
) {
  *diff = lmat - rmat;
}

// fn add(
//   mut sum : &mut M,
//   lmat : &M,
//   rmat : &M
// ) {
//   *sum = lmat + rmat;
// }

