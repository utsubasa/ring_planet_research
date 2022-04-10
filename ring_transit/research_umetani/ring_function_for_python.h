#include "int_ring.h"
#include "params_def.h"


#include <math.h>


/// get eccentric anomaly from M and e
double get_E(double M, double e){
    double E = M;
    double eps = pow(10.0, -7);
    while(fabs(E -e *sin(E)-M) >eps){

        E = E - (E- e*sin(E)-M)/(1.0 - e*cos(E));
    }
    return E;
}

/// get true anomaly from M and e
double get_f(double M , double e){
    double E = get_E(M ,e);
    double f;
    E = E - (int)(E/(2 * M_PI)) * 2 * M_PI;
    //printf("%f %f\n",M,E);
    
    if( 0 < E && E < M_PI){
        f = acos((cos(E) - e)/(1-e*cos(E)));
    }else{
    
        f = 2 * M_PI - acos(( cos(E)-e)/(1 - e * cos(E)));
    }
    
    return f;
}

/// get mean anomaly from e and omega
double mean_ano(double e, double omega){
    return atan2(sqrt(1.0-e*e)*sin((M_PI/2.0) - omega), e + cos((M_PI/2.0) - omega)) - e * sin(atan2(sqrt(1.0-e*e)*sin((M_PI/2.0) - omega), e + cos((M_PI/2.0) - omega)));

}

/// get poistion from time and orbital parameters
void position(double *ppos, double Period, double t_0, double t, double e, double omega, double i, double a){
    double pos[3];
    double n = 2 *M_PI / Period;
    double M = n*(t - (t_0 - ( mean_ano(e, omega)/n) ));
    if( e==0)   M = 2*M_PI * (Period/4.0)/Period  + (t-t_0)*n;
    double E = get_E(M,e);
    //// Omega = 180
    ppos[0] = -(a*(cos(E) - e) * cos (omega) - a * sqrt( 1 -e*e ) *sin(E) * sin(omega) );
    //printf("%.15f %.15f\n", sin(omega), cos(i));
    ppos[1] = -(a * (cos(E) - e ) * sin(omega) + a * sqrt( 1 -e*e ) *sin(E) * cos(omega)) * cos (i);
    ppos[2] = -(a * (cos(E) - e ) * sin(omega) + a * sqrt( 1 -e*e ) *sin(E) * cos(omega)) *sin(i);
    //printf("%f %f %.15f %.15f\n",t,M, ppos[0],ppos[1]);
    
}

// from time and ring, and orbital parameters, we get relative intensity
double ring_and_motion(double t, double period, double t_0, double e, double omega, double i ,double a,  double theta, double phi, double tau, double rp_in, double rp_out, double rp,double r_star, double u1, double u2) {
    double ppos[3];
    position(ppos, period, t_0, t, e, omega, i, a);
    double ring_flux = ring_all(ppos, theta, phi, tau, rp_in, rp_out, rp, r_star, u1,u2);
    double flux_star = M_PI * pow(r_star,2.0) * ( 1- (u1/3.0) - ( u2/6.0)) ;
    
    return (flux_star - ring_flux)/flux_star;
}

void get_flux(double *times, double *p, int datanum, double *fluxes) {
  double porb = p[_period];
  double t_0 = p[_t_0];
  double ecosw = p[_ecosw];
  double esinw = p[_esinw];
  double e, omega;
  if (ecosw==0.0 && esinw==0.0) {
    e = 0.0; omega = 0.0;
  } else {
    e = sqrt(ecosw*ecosw+esinw*esinw); omega = atan2(esinw, ecosw);
  }
  double a_rs = p[_a_rs];
  double b = p[_b];
  double i = acos(b/a_rs);
  double theta = p[_theta];
  double phi = p[_phi];
  double tau = p[_tau];
  double rp_in = p[_rp_in_rp] * p[_rp_rs];
  double rp_out = p[_ratio_out_in] * p[_rp_rs] * p[_rp_in_rp] ;
  double rp = p[_rp_rs];
  double q1 =p[_q1];
  double q2 = p[_q2];
  double u1 = 2 * q2 * sqrt(q1);
  double u2 = sqrt(q1) * (1- 2 * q2);
  double r_star = 1.0;

  gsl_set_error_handler_off();

  int j;
  for (j=0; j<datanum; j++) 
    fluxes[j] = ring_and_motion(times[j], porb, t_0, e, omega, i, a_rs,
				theta, phi, tau, rp_in, rp_out, rp, r_star,
				u1, u2);
}
  
 


