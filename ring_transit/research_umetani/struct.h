#ifndef INCLUDED_STRUCT_H_
#define INCLUDED_STRUCT_H_
/// USEFUL FUNCTIONS


//// INTEGRATION MODULE

typedef struct{ double x_p; double y_p;
 double r_ec; double r_p;
 double r_star; double flat;
  double u1; double u2;
  int flag; double *data;} Params ;

Params init_para( double x_p, double y_p,double r_ec, double r_p,double r_star, double flat,double u1, double u2, int flag){
    Params test;
    test.x_p = x_p;
    test.y_p = y_p;
    test.r_ec = r_ec;
    test.r_p = r_p;
    test.r_star = r_star;
    test.flat = flat;
    test.u1 = u1;
    test.u2 = u2;
    test.flag = flag;

    return test;
}

Params flag_change(Params test, int flag){
    Params test2 = test;
    test2.flag = flag;
    return test2;
}



//// OUTPUT OF GEOMETRIC CONFIGURATION

void dat_make(char *text,  double xp, double yp, double r_star, double r_ec, double r_p, double flat ) {
    double d_theta = 0.0001;
    FILE *fp = fopen(text,"w");
    int i;
    for( i=0; i<100000;i++){
       
        if ( d_theta * i >(3.1415926535) *2)break;
        double theta = i * d_theta;
        //r_star  =r_star;
        
        double x_dai = r_star * cos(theta);
        double y_dai = r_star * sin(theta);
        double x_elli_syo = xp + r_ec * cos(theta);
        double y_elli_syo =yp + r_ec*(1-flat) * sin(theta);
        double x_syo = xp + r_p *cos(theta);
        double y_syo = yp + r_p * sin(theta);
        fprintf(fp,"%.15f %.15f \n",x_dai,y_dai);
        fprintf(fp,"%.15f %.15f \n",x_elli_syo,y_elli_syo);
        fprintf(fp,"%.15f %.15f \n",x_syo,y_syo);


    }
    fclose(fp);
    
}

    
#endif