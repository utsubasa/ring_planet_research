#include "int_ring.h"
#include "params_def.h"
#include "mpfit.h"


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
    ppos[1] = -(a * (cos(E) - e ) * sin(omega) + a * sqrt( 1 -e*e ) *sin(E) * cos(omega)) * cos (i);
    ppos[2] = -(a * (cos(E) - e ) * sin(omega) + a * sqrt( 1 -e*e ) *sin(E) * cos(omega)) *sin(i);

    
}

// from time and ring, and orbital parameters, we get relative intensity
double ring_and_motion(double t, double period, double t_0, double e, double omega, double i ,double a,  double theta, double phi, double tau, double rp_in, double rp_out, double rp,double r_star, double u1, double u2,double norm, double norm2, double norm3, double norm4, double norm5){
    double ppos[3];
    position(ppos, period, t_0, t, e, omega, i, a);
    double ring_flux = ring_all(ppos, theta, phi, tau, rp_in, rp_out, rp, r_star, u1,u2);
    double flux_star = M_PI * pow(r_star,2.0) * ( 1- (u1/3.0) - ( u2/6.0)) ;
    
    return (norm + norm2 * t + norm3 * pow(t,2.0) + norm4 * pow(t,3.0) + norm5 * pow(t,4.0))  * (flux_star - ring_flux)/flux_star;
}

/// this is a ring function for mpfit. this will return relative intensity using the above function
double func_ring(double t, double *p){
    double period = p[_period];
    double t_0 = p[_t_0];
    double e = sqrt( pow(p[_ecosw],2.0) + pow(p[_esinw],2.0));
    double omega;
    if ( p[_esinw] ==0 && p[_ecosw] ==0){
        omega = 0.0;
    }
    else{omega = atan2(p[_esinw],p[_ecosw]);
    }
    double b = p[_b];
    double i = acos(p[_b]/p[_a_rs]);
    double a_rs = p[_a_rs];
    double theta = p[_theta];
    double phi = p[_phi];
    double tau = p[_tau];
    double rp_in_rp = p[_rp_in_rp] * p[_rp_rs];
    double rp_out_rp = p[_ratio_out_in] * p[_rp_rs] * p[_rp_in_rp] ;
    double rp_rs = p[_rp_rs];
    double q1 =p[_q1];
    double q2 = p[_q2];
    double u1 = 2 * q2 * sqrt(q1);
    double u2 = sqrt(q1) * (1- 2 * q2);
    double norm = p[_norm];
    double norm2 = p[_norm2];
    double norm3 = p[_norm3];
    double norm4 = p[_norm4];
    double norm5 = p[_norm5];
    
    double r_star = 1.0;
    int j;

    return ring_and_motion(t,period, t_0, e, omega, i,a_rs,theta,phi,tau, rp_in_rp, rp_out_rp, rp_rs, r_star, u1, u2, norm, norm2, norm3, norm4, norm5);
}

struct mpfit_data{
    double *x;
    double *y;
    double *y_error;
};

/// func_ring(x[i],p) for lmpfit
int ring_lmpfit( int m, int n, double *p, double *deviates, double **derivs, void *private){
    struct mpfit_data *private_2 = (struct mpfit_data *) private;
    int i;
    double *x, *y, *y_error;
    
    x = private_2->x;
    y = private_2->y;
    y_error = private_2->y_error;
    
    for ( i =0 ;i<m; i++) {
        deviates[i] = ( y[i] - func_ring(x[i],p))/y_error[i];
    }
    
    return 0;
}




////~~~~~~~~~~  MONTE CARLO INTEGRATION FOR COMPARING THE TWO METHODS
double _star_limb(double I_0, double u_1, double u_2, double mu){
   return I_0* ( 1- u_1*(1-mu) - u_2 * pow( 1-mu ,2.0) ) ;
   
}

double ring(double R_in, double R_out, double theta, double phi, double tau, double x,double z, double x_p, double z_p, double R_p, double I_0, double R_s, double u_1, double u_2){
    
    double beta = fabs(sin(theta) * cos(phi));
    
    if( pow(x,2.0) + pow(z,2.0) >=pow(R_s,2.0) ) return 0;
    double mu = sqrt( 1 - ((pow(x,2.0) + pow(z,2.0))/pow(R_s, 2.0)));
    
    if( pow(x-x_p,2.0) + pow(z-z_p,2.0) <= pow(R_p,2.0)) return 0;
    
    if( pow(R_in,2.0) < pow( (x-x_p-(z-z_p)*sin(phi)*(1/tan(theta)))/cos(phi), 2.0)+pow((z-z_p)/sin(theta),2.0) && pow( (x-x_p-(z-z_p)*sin(phi)*(1/tan(theta)))/cos(phi), 2.0)+pow((z-z_p)/sin(theta),2.0) < pow( R_out,2.0 ) && pow(x,2.0)  + pow( z, 2.0) < pow(R_s ,2.0)  && pow(x-x_p,2.0) + pow(z-z_p,2.0) > pow(R_p, 2.0) ) return exp(-tau/beta)* _star_limb(I_0, u_1, u_2, mu);
    else return  _star_limb(I_0, u_1, u_2, mu);
    
}

double ring_int(double R_in, double R_out, double theta, double phi, double tau, double x_p, double z_p, double R_p, double I_0, double R_s, double u_1, double u_2,int n_int){
    double d_width = 2*R_out/(double)n_int;
    double x,z;
    int i,j;
    double subtra =0;
    double beta = fabs(sin(theta) * cos(phi));
    for( i=0; i<n_int;i++){
        for( j=0;j<n_int ; j++){
            x = x_p - R_out + d_width * (i+0.5);
            z = z_p - R_out + d_width *  (j +0.5);
            if( pow(x,2.0) + pow(z,2.0)  >= pow( R_s,2.0) )continue;
            double mu = sqrt( 1 - ((pow(x,2.0) + pow(z,2.0))/pow(R_s, 2.0)));
            subtra +=( _star_limb(I_0,u_1,u_2,mu) - ring(R_in, R_out, theta,  phi,  tau,  x, z,  x_p,  z_p,  R_p,  I_0,  R_s,  u_1,  u_2)) * d_width*d_width;
        }
    }
    
    double ring_flux = subtra ;
    
    double flux_star = M_PI * pow(R_s,2.0) * ( 1- (u_1/3.0) - ( u_2/6.0)) ;
    
    return I_0 * (flux_star - ring_flux)/flux_star;
}
//~~~~~~~~~~~


