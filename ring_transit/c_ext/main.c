#include "ring_function.h"
#include <time.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#define dosuu M_PI/180.0
// gcc main.c mpfit.c -lgsl -lgslcblas



double Uniform( void ){
    return ((double)rand()+1.0)/((double)RAND_MAX+2.0);
}

double range_rad(double a, double b){
    return a + Uniform() * (b-a);
}

void copy(double *a, double *b, int NUM_c){
    int i;
    for( i=0 ; i< NUM_c; i++){
        a[i] = b[i];
    }
}

double scale_base(double a, double b, double c, double d, double e, double x){
    return a + x*b + x*x*c + x*x*x*d + x*x*x*x * e;
}


void para_rist(int para_num, mp_par *pars, int flag, double a, double b){
    pars[para_num].fixed = flag;
    if ( flag==0)flag =1;
    else flag = 0;
    pars[para_num].limited[0] =  flag;
    pars[para_num].limited[1] = flag;
    pars[para_num].limits[0] = a;
    pars[para_num].limits[1] = b;
}
int main(void){
 
    clock_t t1, t2;
    t1 = clock();
    srand((unsigned)time(NULL));

  

    /// usefule parameter
    double t;
    int i;


    // data storage
    double x[50000];
    double y[50000];
    double y_2[50000];
    double y_error[50000];

    // making the parameter array
    double p[NPAR];
    double p_2[NPAR];


    /// input file
    FILE *fp = fopen("./data/Q8_l_KIC10403228_test_detrend.dat","r");
    double dmy_x, dmy_y, dmy_error;
    int data_num = 0;
    double l,m,n;
    while ( fscanf(fp,"%lf %lf %lf %lf %lf", &dmy_x, &l,&m ,&dmy_y,&dmy_error ) != EOF){
        x[data_num] = dmy_x;
        y[data_num] = l;
        y_error[data_num] = m;
        data_num++;
    }
    fclose(fp);
    
    // output file
    fp = fopen("./result/fit_result_ring.dat","w");
    
    /// Without this, gsl will stop for because of very small numerical error in integration
    gsl_set_error_handler_off();

    /// initializing the parameter information
    mp_par pars[NPAR];

    /// initializing the result of fitting
    mp_result result;

    /// first we require all parameters to be fixed dufing fitting as as an initial condition
    for ( i =0; i<NPAR; i++){
        pars[i].fixed = 1;
    }

    /// the width of the step when calculating the differentiation
    //pars[_phi].step = 1e-5;


    // reading the parameter file which contains the range of parameters for fitting

    FILE *fp_read = fopen("./data/para_set_ring.txt", "r");
    if( fp_read == NULL){
        printf("ouch!!! no file.\n");
        return -1;
    }
    int count,flag;
    double dmy1, dmy2, dmy3;
    while ( fscanf(fp_read, "%d %lf %d %lf %lf",&count, &dmy1, &flag, &dmy2, &dmy3) != EOF){
        p[count] = dmy1;
        p_2[count] = dmy1;
        para_rist(count, pars, flag, dmy2, dmy3);
        //printf("%d %f %d %f %f\n", count, dmy1, flag, dmy2, dmy3);
    }


    int ring_dof = 0;
    /// print the pars value
    for ( i =0; i < NPAR; i++){
    ring_dof += (1 - pars[i].fixed);
        //printf("%d %d %d %d\n", i, pars[i].fixed, pars[i].limited[0], pars[i].limited[1]);
    }

    // inseting the generated light curve data into arrays
    struct mpfit_data DATA;
    DATA.x = x;
    DATA.y = y;
    DATA.y_error = y_error;

    // status for mpfitting
    int status;

    // arrays for storaging the result of fitting
    double residual[50000];
    double para_error[NPAR];

    // passing the pointer to the "result"
    result.xerror = para_error;
    result.resid = residual;

    // main body of "mpfit" for fitting
    status = mpfit(ring_lmpfit, data_num, NPAR, p, pars, 0, (void *)&DATA, &result);
    printf("nfev:%d niter%d orig_chi:%f best_chi:%f \n",result.nfev, result.niter, result.orignorm, result.bestnorm);

    /// calculating the fitting light curve and outputing the result file which contains
    double scale;
    for ( i =0; i<data_num;i++){
    
        /// y_2[i] = M(t) = polynomial (or scale) * I(t) best-fit
        y_2[i] = func_ring(x[i], p);
        double now[3];
        double incli = acos(p[_b]/p[_a_rs]);
        double e = sqrt( pow(p[_ecosw],2.0) + pow(p[_esinw],2.0));
        double omega = atan2(p[_esinw],p[_ecosw]);
        
        /// M(t) = scale * I(t)
        /// we fit the data to M(t). I(t) = M(t)/scale
        scale = scale_base(p[_norm],p[_norm2],p[_norm3],p[_norm4],p[_norm5],x[i]);
        position(now, p[_period], p[_t_0], x[i], e, omega, incli, p[_a_rs]);
        fprintf(fp,"%.15f %.15f %.15f %.15f %.15f %.15f %.15f\n", x[i],y[i], y_2[i],y_2[i]/scale, y[i]-y_2[i], residual[i], y_error[i] );
        //if( i>39 && i%10 ==0 && i<120){dat_file_inst(fp_dat, now[0], now[1], 1.0, p[_rp_in_rp] * p[_rp_rs], p[_ratio_out_in] * p[_rp_rs] * p[_rp_in_rp] , p[_rp_rs], p[_theta], p[_phi] );
        //printf("%f %f\n", x[i], y_2[i]);
    }
        
    
    // output of the best-fit illustration
    FILE *fp_dat = fopen("./result/ring_geo_ring.dat","w");
  // double time_dat[6] = {-0.75, -0.45, -0.15, 0.15, 0.45, 0.75};
    double time_dat[3] = {p[_t_0]-1, p[_t_0],p[_t_0]+1};
    for ( i=0; i< 3 ;i++){
        double now[3];
        double incli = acos(p[_b]/p[_a_rs]);
        double e = sqrt( pow(p[_ecosw],2.0) + pow(p[_esinw],2.0));
        double omega = atan2(p[_esinw],p[_ecosw]);
        position(now, p[_period], p[_t_0], time_dat[i], e, omega, incli, p[_a_rs]);
        dat_file_inst(fp_dat, now[0], now[1], 1.0, p[_rp_in_rp] * p[_rp_rs], p[_ratio_out_in] * p[_rp_rs] * p[_rp_in_rp] , p[_rp_rs], p[_theta], p[_phi] );
    }

    fclose(fp_dat);


    /// output of the fitting parameters
    FILE *para_file;
    para_file = fopen("./result/para_result_ring.dat","w");
    
    for (i =0; i< NPAR; i++){
        fprintf(para_file, "%d %f %f  %f \n",i, p[i],p_2[i],para_error[i]);
    }
    


    /// output of the statistics
    fprintf(para_file,"%d %d %f %f\n",NPAR,data_num,result.bestnorm,result.bestnorm/(data_num-ring_dof-1));


    /// conputation time
    t2 = clock();
    printf("#%d time %f s", data_num, (double)(t2 - t1) / CLOCKS_PER_SEC );
    //fprintf(para_file,"#%d time %f s", data_num, (double)(t2 - t1) / CLOCKS_PER_SEC );

    fclose(fp);
    fclose(para_file);
    
    return 0;
}