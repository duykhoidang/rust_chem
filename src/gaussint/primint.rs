//File for primitive gaussian evaluations

//integrals

use std::f64::consts::PI;
use rgsl::hypergeometric::hyperg_1F1;

//primitives
pub fn overlap(   //primitive overlap
  z_a : f64,      //exponent for A
  x_a : &[f64;3], //center of A
  i : &[i16;3],   //ang mom of cart directions of A
  z_b : f64,      //exponent for B
  x_b : &[f64;3], //center of B
  j : &[i16;3]    //ang mom of cart directions of B
) -> f64 {
  let mut sum = (PI/(z_a+z_b)).powf(1.5);
  for n in 0..3 {
    let xab = x_a[n] - x_b[n];
    sum *= overlap_x(
      z_a, z_b, xab, i[n], j[n], 0
    );
  }
  return sum;
}

fn overlap_x( //cart component of prim overlap
  z_a : f64,
  z_b : f64,
  xab : f64,
  i : i16,
  j : i16,
  t : i16
) -> f64 {
  let mu = z_a*z_b/(z_a+z_b);
  let p = z_a + z_b;
  // let px = ((z_a*x_a) + (z_b*x_b))/p;
  let mut sum : f64;

  if t > i + j || t < 0 || i < 0 || j < 0 {
    sum = 0.;
  } else if i == 0 && j == 0 && t == 0 {
    sum = (-mu*xab*xab).exp();
  } else if j == 0 { //decrementing i
    sum = (1./(2.*p)) * overlap_x(
      z_a, z_b, xab, i-1, j, t-1
    );
    sum -= (mu*xab/z_a) * overlap_x(
      z_a, z_b, xab, i-1, j, t
    );
    sum += ((t+1) as f64) * overlap_x(
      z_a, z_b, xab, i-1, j, t+1
    );
  } else { //decrementing j
    sum = (1./(2.*p)) * overlap_x(
      z_a, z_b, xab, i, j-1, t-1
    );
    sum += (mu*xab/z_b) * overlap_x(
      z_a, z_b, xab, i, j-1, t
    );
    sum += ((t+1) as f64) * overlap_x(
      z_a, z_b, xab, i, j-1, t+1
    );
  }

  return sum;
}

pub fn ke_int ( //currently not normalized
  z_a : f64,
  x_a : &[f64;3],
  i : &[i16;3],
  z_b : f64,
  x_b : &[f64;3],
  j : &[i16;3] 
) -> f64 {
  let mut sum = 0f64;

  let j1 = j[0];
  let j2 = j[1];
  let j3 = j[2];

  //<0|0>
  sum += z_b * ((2 * (j1+j2+j3) + 3) as f64)
    * overlap(z_a,&x_a,&i,z_b,&x_b,&j);
  //<0|+2> terms
  sum -= 2. * z_a * z_a * (
    overlap(
      z_a,&x_a,&i,z_b,&x_b,&[j1+2,j2,j3]
    ) + 
    overlap(
      z_a,&x_a,&i,z_b,&x_b,&[j1,j2+2,j3]
    ) +
    overlap(
      z_a,&x_a,&i,z_b,&x_b,&[j1,j2,j3+2]
    )
  );
  //<0|-2> terms
  sum -= 0.5 * (
    (j1 * (j1-1)) as f64 * overlap(
      z_a,&x_a,&i,z_b,&x_b,&[j1-2,j2,j3]
    ) + 
    (j2 * (j2-1)) as f64 * overlap(
      z_a,&x_a,&i,z_b,&x_b,&[j1,j2-2,j3]
    ) +
    (j3 * (j3-1)) as f64 * overlap(
      z_a,&x_a,&i,z_b,&x_b,&[j1,j2,j3-2]
    )
  );
  return sum;
}

pub fn nuc_e(
  z_a : f64,
  x_a : &[f64;3],
  i : &[i16;3],
  z_b : f64,
  x_b : &[f64;3],
  j : &[i16;3],
  c : &[f64;3]
) -> f64 {
  let mut sum = 0f64;
  let p = z_a + z_b;
  let pcx = ((z_a*x_a[0]) + (z_b*x_b[0]))/p - c[0];
  let pcy = ((z_a*x_a[1]) + (z_b*x_b[1]))/p - c[1];
  let pcz = ((z_a*x_a[2]) + (z_b*x_b[2]))/p - c[2];
  let rpc = (pcx*pcx + pcy*pcy + pcz*pcz).sqrt();

  let xab = x_a[0] - x_b[0];
  let yab = x_a[1] - x_b[1];
  let zab = x_a[2] - x_b[2];

  let i1 = i[0]; let i2 = i[1]; let i3 = i[2];
  let j1 = j[0]; let j2 = j[1]; let j3 = j[2];

  for t in 0..(i1+j1+1) {
    for u in 0..(i2+j2+1) {
      for v in 0..(i3+j3+1) {
        sum += overlap_x(z_a,z_b,xab,i1,j1,t) *
          overlap_x(z_a,z_b,yab,i2,j2,u) *
          overlap_x(z_a,z_b,zab,i3,j3,v) * 
          rn_tuv(t,u,v,0,p,pcx,pcy,pcz,rpc);
      }
    }
  }
  sum *= 2. * PI/p;
  return sum;
}

fn rn_tuv(
  t : i16,
  u : i16,
  v : i16,
  n : i16,
  p : f64,
  pcx : f64,
  pcy : f64,
  pcz : f64,
  rpc : f64
) -> f64 {
  #[allow(non_snake_case)]
  let T = p * rpc * rpc;
  let mut sum = 0f64;

  if t == 0 && u == 0 && v == 0 {
    sum += (-2. * p).powi(n as i32) * boys(n as f64,T);
  } else if t == 0 && u == 0 {
    if v > 1 {
      sum += ((v-1) as f64) * rn_tuv(t,u,v-2,n+1,p,pcx,pcy,pcz,rpc);
    }
    sum += pcz * rn_tuv(t,u,v-1,n+1,p,pcx,pcy,pcz,rpc);
  } else if t == 0 {
    if u > 1 {
      sum += ((u-1) as f64) * rn_tuv(t,u-2,v,n+1,p,pcx,pcy,pcz,rpc);
    }
    sum += pcy * rn_tuv(t,u-1,v,n+1,p,pcx,pcy,pcz,rpc);
  } else {
    if t > 1 {
      sum += ((t-1) as f64) * rn_tuv(t-2,u,v,n+1,p,pcx,pcy,pcz,rpc);
    }
    sum += pcx * rn_tuv(t-1,u,v,n+1,p,pcx,pcy,pcz,rpc);
  }
  return sum;
}

fn boys(
  n : f64,
  t : f64
) -> f64 {
  hyperg_1F1(n+0.5,n+1.5,-t)/(2.*n+1.)
}

pub fn eri_prim(
  z_a : f64,
  x_a : &[f64;3],
  i : &[i16;3],
  z_b : f64,
  x_b : &[f64;3],
  j : &[i16;3],
  z_c : f64,
  x_c : &[f64;3],
  k : &[i16;3],
  z_d : f64,
  x_d : &[f64;3],
  l : &[i16;3],
) -> f64 {
  let mut sum = 0f64;
  let i1 = i[0]; let i2 = i[1]; let i3 = i[2];
  let j1 = j[0]; let j2 = j[1]; let j3 = j[2];
  let k1 = k[0]; let k2 = k[1]; let k3 = k[2];
  let l1 = l[0]; let l2 = l[1]; let l3 = l[2];
  
  let p = z_a+z_b; //(z_a+z_b);
  let q = z_c+z_d; //(z_c+z_d);
  let alpha = p*q/(p+q);

  let px = (z_a*x_a[0] + z_b*x_b[0])/p;
  let py = (z_a*x_a[1] + z_b*x_b[1])/p;
  let pz = (z_a*x_a[2] + z_b*x_b[2])/p;

  let qx = (z_c*x_c[0] + z_d*x_d[0])/q;
  let qy = (z_c*x_c[1] + z_d*x_d[1])/q;
  let qz = (z_c*x_c[2] + z_d*x_d[2])/q;

  let xab = x_a[0]-x_b[0]; 
  let yab = x_a[1]-x_b[1]; 
  let zab = x_a[2]-x_b[2];
  
  let xcd = x_c[0]-x_d[0]; 
  let ycd = x_c[1]-x_d[1]; 
  let zcd = x_c[2]-x_d[2];

  let rpq = ((px-qx).powi(2) +
      (py-qy).powi(2) +
      (pz-qz).powi(2)   
    ).sqrt();
  
  for t1 in 0..(i1+j1+1) {
    for u1 in 0..(i2+j2+1) {
      for v1 in 0..(i3+j3+1) {
        for t2 in 0..(k1+l1+1) {
          for u2 in 0..(k2+l2+1) {
            for v2 in 0..(k3+l3+1) {
              sum += overlap_x(z_a,z_b,xab,i1,j1,t1) *
                overlap_x(z_a,z_b,yab,i2,j2,u1) *
                overlap_x(z_a,z_b,zab,i3,j3,v1) *
                overlap_x(z_c,z_d, xcd,k1,l1,t2) *
                overlap_x(z_c,z_d, ycd,k2,l2,u2) *
                overlap_x(z_c,z_d, zcd,k3,l3,v2) *
                ((-1i16).pow((t2+u2+v2) as u32) as f64) *
                rn_tuv(t1+t2,u1+u2,v1+v2,0,alpha,
                  px-qx,py-qy,pz-qz,rpq
                );
            }
          }
        }
      }
    }
  }

  sum *= 2.*PI.powf(2.5)/(p*q*((p+q).sqrt()));

  return sum;
}