#ifndef INCLUDED_INT_RING_H_
#define INCLUDED_INT_RING_H_

#include <gsl/gsl_poly.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>

#include "struct.h"
#include "ellipse_circle.h"

#define epsrel 1.0e-4
#define epsabs 0.0


// intensity function  for decart coordinate  where the center is at the centert of the planet
// posi is (x + xp, y + yp);
double get_intensity_decart(const double y, void *p){
    Params * params = (Params *)p;
    double xp = params->x_p;
    double yp = params->y_p;
    double x = params->data[1];
    double r_star = params->r_star;
    double u1 = params->u1;
    double u2 = params->u2;
    

    double r = sqrt( pow(x+xp,2.0) + pow( y + yp,2.0) ) ;
    if( r >  r_star ) return 0;
    double mu = sqrt( 1 - pow(r/r_star, 2.0));
    return 1 - u1 * ( 1-mu ) - u2 * pow(1 -mu,2.0) ;
}


// ~~~~~~~~~~~~     integral function for FULL transit of ellipse ( if flat =0, this function gives integration of a circle
double flux_int_int(const double x, void *p){
    Params * params = (Params *)p;
    double r = params->data[0];
    double flat = params->flat;

    double y_max, y_min;
    params->data[1] = x;
    int i;
    double ret,ret_abserr;
    size_t n_eval;
    if( params->r_p == params->data[0])flat =0.0;
  
    y_max = (1-flat ) * sqrt ( pow(r,2.0) - pow(x,2.0) );
    y_min = -(1-flat) * sqrt ( pow(r,2.0) - pow(x,2.0) );
  
    gsl_function f;
    f.function = &get_intensity_decart;
    f.params = params;
  
    gsl_integration_qng(&f,y_min, y_max, epsabs, epsrel, &ret, &ret_abserr, &n_eval);
  
  
    return ret;
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



/// integration of the FULL transit using Decartes coordinate
double flux_intfull(void *p){
    Params * params = (Params *)p;
    double data[2];
    params->data = data;
    //printf("FULX FULL");

    //// setting the parameteres ?????
    double r_p = params->r_p;
    double r_ec = params->r_ec;
    double flat = params->flat;
    
    int flag = params->flag;  /// flag let you determine a style of an integration
               /// 1: elliptic
               /// 2: elliptic and circle
    /// ~~~~~~~~~~~~~~~~~~~~~~~
   
    
    gsl_function f;
    f.function = &flux_int_int;
    f.params = params;
    double ret,ret_abserr;
    size_t n_eval;

    double sum =0.0;
    double x_min; double x_max;

    switch(flag){
        /////  for ellipse
        case 1:
            data[0] = r_ec;
            x_min = -r_ec;
            x_max = r_ec;
            gsl_integration_qng(&f, x_min, x_max, epsabs, epsrel, &ret, &ret_abserr, &n_eval);
            //printf("ret %f\n", ret);

            return ret;
        
        
        ///// for ellipse and circle
        case 2:
        
            
            ///circle integration
            x_min = -r_p;
            x_max = - sqrt(  (pow(r_p,2.0)  - pow((1-flat) * r_ec ,2.0) )/( 2 *flat - pow(flat,2.0)) ) ;
            data[0] = r_p;
            f.params = params;
            gsl_integration_qng(&f, x_min, x_max, epsabs, epsrel, &ret, &ret_abserr, &n_eval);
            sum += ret;
           // printf("ret %f\n", ret);

            
            /////  ellipse integration
            data[0] = r_ec;
            
            x_min =  - sqrt(  (pow(r_p,2.0)  - pow((1-flat) * r_ec ,2.0) )/( 2 *flat - pow(flat,2.0)) ) ;
            x_max = sqrt(  (pow(r_p,2.0)  - pow((1-flat) * r_ec ,2.0) )/( 2 *flat - pow(flat,2.0)) ) ;
            f.params = params;
            gsl_integration_qng(&f, x_min, x_max, epsabs, epsrel, &ret, &ret_abserr, &n_eval);
            sum += ret;
            //printf("ret %f\n", ret);

            
            
            /// circle integration
            data[0] = r_p;
            x_min =  sqrt(  (pow(r_p,2.0)  - pow((1-flat) * r_ec ,2.0) )/( 2 *flat - pow(flat,2.0)) ) ;
            x_max = r_p;
            f.params = params;
            gsl_integration_qng(&f, x_min, x_max, epsabs, epsrel, &ret, &ret_abserr, &n_eval);
            sum += ret;
            //printf("ret %f\n", ret);

            return sum;
        
        
        
        default:
            printf("Please enter an appropriate command ");
    }
return 0;
}
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// intensity function  for polar cooridante
double get_intensity_polar(const double r, void *p){
    Params * params = (Params *)p;


    double mu = sqrt( 1 - pow(r/params->r_star, 2.0));
    double here =  1 - params->u1 * ( 1-mu ) - params->u2 * pow(1 -mu,2.0) ;
    if(here<0) printf("wowowow\n");
    return here;
}




double flux_thetaint(const double r, void *p){
    Params * params = (Params *)p;

    double theta[4];////  theta[0] ~ theta[1] or theta[0] ~ theta[1], theta[2] ~ theta[3]
    int num_theta; // the number of the theta solutions
    
    double ppos[3];
    ppos[0] = params->x_p;
    ppos[1] = params->y_p;
    
    if( params->flag ==1 ){
        if( params->flat != 0 )theta_ext_ellipse(ppos,params->r_ec, params->flat, r, theta, &num_theta);
        else theta_set_circle(ppos, params->r_p, r, theta, &num_theta);
    }
    else if( params->flag ==2 ) theta_ext_ellipse_circle( ppos, params->r_ec, params->r_p, params->flat, r, theta, &num_theta);
    
    double data = r;
    params->data = &data;

    int i;

    gsl_function f;
    f.function = &get_intensity_polar;
    f.params = params;
    
    double ret1, ret2, ret_abserr, ret_abserr2;
    size_t n_eval, n_eval2;
    
    double intensity = get_intensity_polar(r,params);
  
    
    /// theta[0] ~ theta[1]
    if ( num_theta == 1){
        return 0;
    }
    if(num_theta == 2){
        if ( theta[1] - theta[0] > M_PI){
        
        ret1 = (theta[0] + 2 * M_PI - theta[1]) * intensity;

        }
        
        else{
        ret1 = (theta[1] - theta[0] ) * intensity;
        }

        return r * ret1;
    }
    
    if ( num_theta == 3){
        for ( i=0; i<3;i++){
        }
        if ( theta[2] - theta[0] > M_PI){
            ret1 = (theta[0] + 2 * M_PI - theta[2]) * intensity;
        }
        else{
            ret1 = (theta[2] - theta[0] ) * intensity;
        }
        

  
        return 0;
    }
    
    /// theta[0] ~ theta[1], theta[2] ~ theta[3]
    if( num_theta == 4){

        if ( theta[1] - theta[0] > M_PI){
            ret1 = (theta[0] + 2 * M_PI - theta[1]) * intensity;


        }
        else{
            ret1 = (theta[1] - theta[0] ) * intensity;

        }
        if ( theta[3] - theta[2] > M_PI){
            ret2 = (theta[2] + 2 * M_PI - theta[3]) * intensity;


        }
        else{
            ret2 = (theta[3] - theta[2] ) * intensity;

        }

        return r * (ret1 + ret2);
        
    }
    else return 0;
    
}



////  the integration function in case of edge
double integral_main(void *p){
    Params * params = (Params *)p;

    double ppos[3];
    ppos[0] = params->x_p;
    ppos[1] = params->y_p;
    double x_res[4];
    double y_res[4];
    double r_ext[10];
    double theta_d[100];
    int r_ext_num;
    double r_star = params->r_star;
    if ( params->flag ==1){
    
        solve_fourth_rext(ppos[0], ppos[1], params->r_ec, params->flat, r_ext, &r_ext_num,x_res, y_res);
        
    }
    
    
    if( params->flag ==2 ){
        set_r_ext_circle_ellipse(ppos, params->flat, params->r_ec, params->r_p, r_ext, &r_ext_num);
    }

    gsl_function f;
    f.function = &flux_thetaint ;
    f.params = params ; ///// define the paramerter

    double ret, ret_abserr ;
    double sum =0.0;
    size_t n_eval ;
    
    int r_star_num=0;
    int i;
    if( r_ext[r_ext_num-1] <= r_star){
        return flux_intfull(params);
    }else{
        
        /// r_ext[0] < r_ext[1] < r_ext[2] ..... <r_ext[r_ext_num-1]
        for ( i =0 ; i < r_ext_num; i++){
          //  printf("i:%d r_Ext:%.15f\n",i,r_ext[i]);
            if ( r_star < r_ext[i] ){
                r_star_num = i;
                break;
            }
        }
        
        /// r_ext[0] < r_ext[1] < ..... < r_star
        r_ext[r_star_num] = r_star;
        for ( i=0; i< r_star_num; i++){
            gsl_integration_qng(&f, r_ext[i], r_ext[i+1], epsabs, epsrel, &ret, &ret_abserr, &n_eval);
            sum += ret;
        }
    }
    return sum;
}



//// y
//// ^
//// |
///  |
//   =====> x
///
//
/// a x^2 + 2b xy + c y^2 = rp^2
/// we derive the angle "theta" between the major axis and x-axis
/// we also derive the length of the major axis "big_rad" and the short axis "small_rad"
int diag_ellipse(double a, double b, double c, double rp, double *theta, double *big_rad, double *small_rad){
    
    if( a == c && b == 0){
        *theta =0.0;
        *big_rad = *small_rad = rp * sqrt(1.0/a);
        return 1;
    }
    
    
    if( b==0){
        if( a< c){
            *theta =0;
            *big_rad = rp* sqrt(1.0/a);
            *small_rad = rp * sqrt( 1.0/c);
        }else{
            *theta = M_PI /2.0;
            *big_rad = rp* sqrt(1.0/c);
            *small_rad = rp * sqrt( 1.0/a);
        }
        return 1;
    }
    
    double big_dig = ((a+c)/2.0) +0.5 * sqrt(pow(a-c,2.0) + 4 * b *b) ;
    double small_dig = ((a+c)/2.0) -0.5 * sqrt(pow(a-c,2.0) + 4 * b *b) ;
    double x  = 1.0;
    double y =(-a+ x * small_dig)/b;
    *theta = atan2(y,x);
    *big_rad = rp * sqrt(1.0/small_dig);
    *small_rad = rp * sqrt(1.0/big_dig);
    
    return 1;
}


//// we calculate all flux occultated by a planet and rings
double ring_all(double *ppos, double theta, double phi, double tau, double rp_in, double rp_out, double rp,double r_star, double u1, double u2){

    /// if there is no ring, we only consider the occultation of the planet
    if( cos(phi) ==0 || sin(theta ) ==0){
        Params test_test = init_para(ppos[0],ppos[1], rp, rp, r_star, 0.0,u1,u2,1);
        return integral_main(&test_test);
    }
    
    /// we calculate the direction and the length of the major and short axis of the in and out ring using diag_ellipse
    double a = 1.0/(pow(cos(phi),2.0));
    double b =  sin(phi) / (pow(cos(phi),2.0)  * tan(theta));
    double c = pow(tan(phi)/tan(theta),2.0) + pow(1/sin(theta),2.0);
    
    double rot_theta;
    double big_rad_in, big_rad_out;
    double small_rad_in, small_rad_out;
    
    
    diag_ellipse(a,b,c,rp_in, &rot_theta, &big_rad_in, &small_rad_in);
    diag_ellipse(a,b,c,rp_out, &rot_theta, &big_rad_out, &small_rad_out);


    //// rotate the system and align the x -axis with the ellipse's principal long axis
    double pos_now[2];
    pos_now[0] = cos(rot_theta) * ppos[0] +sin(rot_theta) * ppos[1];
    pos_now[1] = -sin(rot_theta) * ppos[0] + cos(rot_theta) * ppos[1];
    
    /*
    typedef struct{ double x_p; double y_p;
    double r_p; double r_ec;
     double r_star; double flat;
      double u1; double u2;
      int flag; double *data;} Params ;
    */
    
    Params in_para_1, out_para_1, circle_para_1;
    Params in_para_2, out_para_2, circle_para_2;

    
    in_para_1 = init_para(pos_now[0], pos_now[1], big_rad_in, rp, r_star, (big_rad_in - small_rad_in)/big_rad_in,u1,u2,1);
    out_para_1 =init_para(pos_now[0],pos_now[1], big_rad_out, rp, r_star, (big_rad_out - small_rad_out)/big_rad_out,u1,u2,1);
    circle_para_1 = init_para(pos_now[0],pos_now[1], rp, rp, r_star, 0.0,u1,u2,1);
    
    in_para_2 = init_para(pos_now[0], pos_now[1], big_rad_in, rp, r_star, (big_rad_in - small_rad_in)/big_rad_in,u1,u2,2);
    out_para_2 =init_para(pos_now[0],pos_now[1], big_rad_out, rp, r_star, (big_rad_out - small_rad_out)/big_rad_out,u1,u2,2);
    circle_para_2 = init_para(pos_now[0],pos_now[1], rp, rp, r_star, 0.0,u1,u2,2);
    
    int integral_type = 0 ;
    if(rp <= small_rad_in)integral_type = 1;
    if(small_rad_in <= rp && rp <= small_rad_out)integral_type = 2;
    if(small_rad_out <= rp)integral_type = 3;
    double beta = fabs(sin(theta) * cos(phi));

    double att = tau;
    
    switch ( integral_type ){
    
        case 1:
            return integral_main(&circle_para_1) + att * integral_main(&out_para_1) - att * integral_main(&in_para_1);
            
        case 2:
            return integral_main(&circle_para_1) * ( 1- att) + att * integral_main(&out_para_1) - att * ( integral_main(&in_para_1) - integral_main(&in_para_2));
        
        case 3:
            return att* ( integral_main(&out_para_1)- integral_main(&out_para_2)) - att * (integral_main(&in_para_1) - integral_main(&in_para_2))+integral_main(&circle_para_1);
        
        default:
            return 0;
    }
    
    return 0;
}


int dat_file_inst(FILE *fp,  double xp, double yp, double r_star, double rp_in, double rp_out, double rp, double theta_ring, double phi_ring) {
    double d_theta = 0.001;
    int i;
    
      if( cos(phi_ring) ==0 || sin(theta_ring ) ==0){
        for( i =0; i< 10000; i++){
            double theta = i*d_theta;
            fprintf(fp,"%.15f %.15f\n", r_star *cos(theta), r_star*sin(theta));
            fprintf(fp,"%.15f %.15f\n", xp + rp*cos(theta), yp + rp*sin(theta));
        }
        return 0;
      }
    
    /// we calculate the direction and the length of the major and short axis of the in and out ring using diag_ellipse
    double a = 1.0/(pow(cos(phi_ring),2.0));
    double b =  sin(phi_ring) / (pow(cos(phi_ring),2.0)  * tan(theta_ring));
    double c = pow(tan(phi_ring)/tan(theta_ring),2.0) + pow(1/sin(theta_ring),2.0);
    
    double rot_theta;
    double big_rad_in, big_rad_out;
    double small_rad_in, small_rad_out;
    
    diag_ellipse(a,b,c,rp_in, &rot_theta, &big_rad_in, &small_rad_in);
    diag_ellipse(a,b,c,rp_out, &rot_theta, &big_rad_out, &small_rad_out);
    
    
    for( i=0; i<10000;i++){
       
        if ( d_theta * i >(3.1415926535) *2)break;
        double theta = i * d_theta;
        //r_star  =r_star;
        double dmy1,dmy2;
        
        double x_dai = r_star * cos(theta);
        double y_dai = r_star * sin(theta);
        dmy1 = big_rad_in * cos(theta);
        dmy2 = small_rad_in * sin(theta);
        double x_elli_syo = xp + cos(rot_theta)*dmy1 -sin(rot_theta) * dmy2;
        double y_elli_syo =yp +  ( sin(rot_theta) * dmy1 + cos(rot_theta) * dmy2);
        dmy1 = big_rad_out * cos(theta);
        dmy2 = small_rad_out * sin(theta);
        double x_elli_dai = xp + cos(rot_theta)*dmy1 -sin(rot_theta) * dmy2;
        double y_elli_dai =yp + ( sin(rot_theta) * dmy1 + cos(rot_theta) * dmy2);
        
        double x_syo = xp + rp *cos(theta);
        double y_syo = yp + rp * sin(theta);
        fprintf(fp,"%.15f %.15f \n",x_dai,y_dai);
        fprintf(fp,"%.15f %.15f \n",x_elli_syo,y_elli_syo);
        fprintf(fp,"%.15f %.15f \n",x_elli_dai,y_elli_dai);
        fprintf(fp,"%.15f %.15f \n",x_syo,y_syo);


    }
    return 0;
}

#endif


