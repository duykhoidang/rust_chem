use super::primint::*;

//contracted gaussian
pub struct Gaussian { 
  pub ngs : u32,
  coord : [f64;3],
  exps : Vec<f64>,
  coeffs : Vec<f64>,
  shell : [i16;3],
}

//Gaussian methods
impl Gaussian {
  pub fn new(
    ngs : u32,
    coord : [f64;3],
    exps : Vec<f64>,
    coeffs : Vec<f64>,
    shell : [i16;3]
  ) -> Gaussian {
    let mut g = Gaussian {
      ngs : ngs,
      coord : coord,
      exps : exps,
      coeffs : coeffs,
      shell : shell
    };
    g.normalize();
    return g;
  }

  fn normalize(&mut self) {
    let norm = Gaussian::get_sij(&self,&self).sqrt();

    for cf in self.coeffs.iter_mut() {
      *cf /= norm;
    }
  }

  pub fn get_sij(
    i : &Gaussian,
    j : &Gaussian
  ) -> f64 {
    let mut sij = 0f64;

    for (cf1,exp1) in i.coeffs.iter().zip(i.exps.iter()) {
      for (cf2,exp2) in j.coeffs.iter().zip(j.exps.iter()) {
        sij += cf1 * cf2 * overlap(
          *exp1, &i.coord, &i.shell,
          *exp2, &j.coord, &j.shell
        );
      }
    }

    return sij;
  }

  pub fn get_tij(
    i : &Gaussian,
    j : &Gaussian
  ) -> f64 {
    let mut tij = 0f64;

    for (cf1,exp1) in i.coeffs.iter().zip(i.exps.iter()) {
      for (cf2,exp2) in j.coeffs.iter().zip(j.exps.iter()) {
        tij += cf1 * cf2 * ke_int(
          *exp1, &i.coord, &i.shell,
          *exp2, &j.coord, &j.shell
        );
      }
    }
    
    return tij;
  }

  pub fn get_ven(
    i : &Gaussian,
    j : &Gaussian,
    c : &Vec<[f64;3]>,
    z : &Vec<i16>
  ) -> f64 {
    let mut vij = 0f64;

    for (cf1,exp1) in i.coeffs.iter().zip(i.exps.iter()) {
      for (cf2,exp2) in j.coeffs.iter().zip(j.exps.iter()) {
        for (ci,zi) in c.iter().zip(z.iter()) {
          vij -= (*zi as f64) * cf1 * cf2 * nuc_e(
            *exp1, &i.coord, &i.shell,
            *exp2, &j.coord, &j.shell, &ci
          );
        }
      }
    }    
  
    return vij;
  }

  pub fn get_eri(
    i : &Gaussian,
    j : &Gaussian,
    k : &Gaussian,
    l : &Gaussian
  ) -> f64 {
    let mut eri = 0f64;
    for (cf1,exp1) in i.coeffs.iter().zip(i.exps.iter()) {
      for (cf2,exp2) in j.coeffs.iter().zip(j.exps.iter()) {
        for (cf3,exp3) in k.coeffs.iter().zip(k.exps.iter()) {
          for (cf4,exp4) in l.coeffs.iter().zip(l.exps.iter()) {
            eri += cf1 * cf2 * cf3 * cf4 * eri_prim(
              *exp1, &i.coord, &i.shell,
              *exp2, &j.coord, &j.shell, 
              *exp3, &k.coord, &k.shell, 
              *exp4, &l.coord, &l.shell
            );
          }
        }
      }
    }
    return eri;
  }
}