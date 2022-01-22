import jaxnumpy as jnp
#MONTE CARLO INTEGRATION FOR COMPARING THE TWO METHODS
def _star_limb(double I_0, double u_1, double u_2, double mu):
   return I_0* ( 1- u_1*(1-mu) - u_2 * jnp.power( 1-mu ,2.0) )

def ring(double R_in, double R_out, double theta, double phi, double tau, double x,double z, double x_p, double z_p, double R_p, double I_0, double R_s, double u_1, double u_2):
    beta = jnp.abs(jnp.sin(theta) * jnp.cos(phi))
    if jnp.power(x,2.0) + jnp.power(z,2.0) >=jnp.power(R_s,2.0):
        return 0;
    mu = jnp.sqrt( 1 - ((jnp.power(x,2.0) + jnp.power(z,2.0))/jnp.power(R_s, 2.0)))
    if jnp.power(x-x_p,2.0) + jnp.power(z-z_p,2.0) <= jnp.power(R_p,2.0)):
        return 0
    if jnp.power(R_in,2.0) < jnp.power( (x-x_p-(z-z_p)*jnp.sin(phi)*(1/jnp.tan(theta)))/jnp.cos(phi), 2.0)+jnp.power((z-z_p)/jnp.sin(theta),2.0) and \
        jnp.power( (x - x_p - (z-z_p) * jnp.sin(phi) * (1/jnp.tan(theta)))/jnp.cos(phi), 2.0) + jnp.power((z-z_p)/sin(theta),2.0) < jnp.power( R_out,2.0 ) and \
        jnp.power(x,2.0) + jnp.power(z, 2.0) < jnp.power(R_s ,2.0)  and\
        jnp.power(x-x_p,2.0) + jnp.power(z-z_p,2.0) > jnp.power(R_p, 2.0) ):
        return jnp.exp(-tau/beta)* _star_limb(I_0, u_1, u_2, mu)
    else:
        return  _star_limb(I_0, u_1, u_2, mu)

def ring_int(R_in, R_out, theta, phi, tau, x_p, z_p, R_p, I_0, R_s, u_1, u_2, n_int, x,z):
    d_width = 2*R_out/n_int
    subtra = 0
    beta = jnp.abs(jnp.sin(theta) * jnp.cos(phi))
    i_max = int((R_s**2 / d_width) -0.5) )
    for i in range(i_max+1):
        for j in n_int:
            x = x_p - R_out + d_width * (i+0.5)
            z = z_p - R_out + d_width *  (j +0.5)
            mu = jnp.sqrt( 1 - ((jnp.power(x,2.0) + jnp.power(z,2.0))/jnp.power(R_s, 2.0)))
            subtra +=( _star_limb(I_0,u_1,u_2,mu) - ring(R_in, R_out, theta,  phi,  tau,  x, z,  x_p,  z_p,  R_p,  I_0,  R_s,  u_1,  u_2)) * d_width*d_width
    ring_flux = subtra
    flux_star = M_PI * jnp.power(R_s,2.0) * ( 1- (u_1/3.0) - ( u_2/6.0))
    return I_0 * (flux_star - ring_flux)/flux_star

'''
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
            if( pow(x,2.0) + pow(z,2.0)  >= pow( R_s,2.0) )continue; //x^2+z^2 >= R_s^2なら処理を中断.これをforループの上限値に設定すればよい。
            double mu = sqrt( 1 - ((pow(x,2.0) + pow(z,2.0))/pow(R_s, 2.0)));
            subtra +=( _star_limb(I_0,u_1,u_2,mu) - ring(R_in, R_out, theta,  phi,  tau,  x, z,  x_p,  z_p,  R_p,  I_0,  R_s,  u_1,  u_2)) * d_width*d_width;
        }
    }
    double ring_flux = subtra ;
    double flux_star = M_PI * pow(R_s,2.0) * ( 1- (u_1/3.0) - ( u_2/6.0)) ;
    return I_0 * (flux_star - ring_flux)/flux_star;
}
//~~~~~~~~~~~
'''
