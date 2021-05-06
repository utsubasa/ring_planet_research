#ifndef INCLUDED_ELLIPSE_CIRCLE_H_
  
#define INCLUDED_ELLIPSE_CIRCLE_H_
#include <gsl/gsl_poly.h>
#include <math.h>
#include <stdio.h>
#include "sort.h"


///  if x,y is inside daen, this returns plus
double daen(double *ppos, double x,double y, double a,double b){

    return 1- pow((ppos[0]-x) /a,2.0) - pow( ( ppos[1]-y) /b,2.0);
}


/// if x, y is inside circle, this returns plus
double circle(double *ppos, double x, double y, double rp){

    return 1 - pow((ppos[0]-x)/rp,2.0) - pow((ppos[1] -y ) /rp,2.0) ;

}

////  we can derive the intersection of an ellipse(  whose radius are a and b) and a circle( whose radius is r_star )
//// ellipse's center is ( ppos[0], ppos[1]) and circle's center is (0,0)
//// we derive the x-coordinate of the intersetion whose center is (ppos[0],ppos[1])
int solve_fourth_intersection(double *ppos, double a,double b,double r_star,double z[8]){
    
    
    double A[4];
    int i;

    A[0] = 1-((b*b)/(a*a));
    A[1] = 2 * ppos[0];
    A[2] = ppos[0] * ppos[0] + ppos[1] * ppos[1] + b*b - r_star * r_star;
    A[3] = - 2 * ppos[1] ;
    
    

    /* P(x) = -1 + x^4 の係数 */
    double coe[5] = { -1, 0, 0, 0, 1 };
    coe[4] = A[0] * A[0];
    coe[3] = 2 * A[0] * A[1];
    coe[2] = 2 * A[0] * A[2] + A[1] * A[1] + (b*b*A[3]*A[3]/(a*a));
    coe[1] = 2 * A[1] * A[2] ;
    coe[0] = A[2] * A[2] - b * b * A[3] * A[3];


    //// if a ==b we change the strantegy (( we don't use this command unless theta = pi/2.0 && phi =0 )
    if( A[0] == 0 ){
        if( ppos[1] ==0){
           // printf("%f\n",( pow( ppos[0],2.0) + pow(r_star,2.0) - pow(a,2.0))/ ( 2*ppos[0]));
            z[0] =  ( pow( ppos[0],2.0) + pow(r_star,2.0) - pow(a,2.0))/ ( 2*ppos[0]) - ppos[0];
            z[1] = 0.0;
            z[2] = ( pow( ppos[0],2.0) + pow(r_star,2.0) - pow(a,2.0))/ ( 2*ppos[0]) - ppos[0];
            z[3] = 0.0;
            z[4] = 0.001;
            z[5] = 1e20;
            z[6] = 0.001;
            z[7] = 1e20;
        }else if(pow(coe[1],2.0) - 4 * coe[2] * coe[0] >0){
            z[0] = ( -coe[1] + sqrt(pow(coe[1],2.0) - 4 * coe[2] * coe[0]) )/(2 * coe[2]);
            z[1] = 0.0;
            z[2] = ( -coe[1] - sqrt(pow(coe[1],2.0) - 4 * coe[2] * coe[0]) )/(2 * coe[2]);
            z[3] = 0.0;
            z[4] = 0.001;
            z[5] = 1e20;
            z[6] = 0.001;
            z[7] = 1e20;
        }else{
            for(i = 0 ;i< 4; i++){
                z[2*i] = 0.0001;
                z[2*i+1] = 1e20;
        }
    }
        return 0;
}
    

    
    // we solve the quad equation and get the x coordinate
    gsl_poly_complex_workspace * w = gsl_poly_complex_workspace_alloc(5);
    gsl_poly_complex_solve(coe, 5, w, z);
    gsl_poly_complex_workspace_free(w);
    
    return 0;
}



//// we can derive extreme values of the distance from the center of the star ( 0,0 ) to the ellipse centered at (xp, yp)
//// we get four solutions at most and two solutions at least

int solve_fourth_rext(double xp,double yp,double rp,double f,double r_ext[4], int *r_num, double *x_res, double *y_res){
    
    double z[8];
    
    /// declaring the paramters for the quad equation
    double c = pow(xp/rp,2.0);
    double d = pow(yp/(rp*(1-f)),2.0);
    double a = pow(1.0/( (1-f) * rp),2.0);
    double b  =pow(1.0/rp,2.0);

    ////  an exapmle of the coefficent
    /* P(x) = -1 + x^4  */
    /// coe are coefficients of quad equation
    double coe[5] = { -1, 0, 0, 0, 1 };

    // inserting the coeffiient for this problem
    coe[4] = pow(a*b,2.0);
    coe[3] = 2 * a*b*(a+b);
    coe[2] = a*a + 4 * a * b + b*b -c*a*a - d*b*b;
    coe[1] = 2*a+2*b - 2*a*c - 2*b*d;
    coe[0] = 1-c-d;




    int i, j;
    //// num_count is a counter of results.
    int num_count;
    
    // if xp is 0, the solution becomes divergence so we need to solve it analytically
    if( xp == 0 ){

        double r = sqrt(pow(xp,2.0) + pow( yp,2.0) );
        if( pow(rp,2.0) - ( pow(1-f,2.0) * pow(yp,2.0) /(2*f -pow(f,2.0)) ) >0 ){
            
            r_ext[0] = r -(1-f) * rp;
            x_res[0] = 0.0;
            y_res[0] =  -(1-f) * rp;
            
            r_ext[1] = r+ ( 1-f) *rp;
            x_res[1] = 0.0;
            y_res[1] =  ( 1-f) *rp;
            
            double dmy_x = sqrt ( pow(rp,2.0) - ( pow(1-f,2.0) * pow(yp,2.0) /(2*f -pow(f,2.0)) ));
            double dmy_y = pow(1-f,2.0) * yp/(2*f -pow(f,2.0));
            
            r_ext[2] = sqrt ( pow(dmy_x,2.0) + pow(yp + dmy_y,2.0)  );
            x_res[2] = dmy_x;
            y_res[2] =  dmy_y;
            
            r_ext[3] = sqrt ( pow(-dmy_x,2.0) + pow(yp + dmy_y,2.0) )  ;
            x_res[3] = -dmy_x;
            y_res[3] =  dmy_y;
            num_count = 4;

            sort_double(r_ext, num_count, sizeof(r_ext[0]));
            *r_num = 4;
            return 0;
        }
        else{

            r_ext[0] = r -(1-f) * rp;
            x_res[0] = 0.0;
            y_res[0] =  -(1-f) * rp;
            
            r_ext[1] = r+ ( 1-f) *rp;
            x_res[1] = 0.0;
            y_res[1] =  ( 1-f) *rp;
            
            num_count = 2;
            sort_double(r_ext, num_count, sizeof(r_ext[0]));

            *r_num = 2;
            
            return 0;
        
        
        }
    }
    
    
  
    
    // solving the quad equation using the gls library
    gsl_poly_complex_workspace * w= gsl_poly_complex_workspace_alloc(5);
    gsl_poly_complex_solve(coe, 5, w, z);
    gsl_poly_complex_workspace_free(w);


    double x[4];
    double y[4];

    num_count =0;
    
    /// checking whether the results are imaginary number  or not
    /// Be careful that under the condition where the origin is centered at the center of the star,
    /// distances from the origin to the ellipse sqrt( pow(x[i] + xp,2.0) + pow(y[i] + yp,2.0))
    for ( i =0; i < 4 ; i++){
        
        if( yp ==0 && (1+(z[2*i]/pow((1-f)*rp,2.0))) < 1e-10){
            x[i] = -xp/(1+(z[2*i]/pow(rp,2.0)));
            y[i] = 0.0;
        }else{
            x[i] = -xp/(1+(z[2*i]/pow(rp,2.0)));
            y[i] = -yp/(1+(z[2*i]/pow((1-f)*rp,2.0)));
        }

        if(fabs(z[2*i+1]/z[2*i]) <1e-6 || z[2*i+1] ==0){

        if( fabs(x[i]) >rp*(1+1e-10)) continue;

        if( fabs(y[i]) > rp*(1+1e-10) ) continue;

        x_res[num_count] = x[i] ;  //?????
        y_res[num_count] = y[i] ;  //????? x and y are not sorted
            
        r_ext[num_count] = sqrt( pow(x[i] + xp,2.0) + pow(y[i] + yp,2.0)) ;
      //  printf("%f %f %f\n",x[i],y[i], r_ext[num_count]);
        
        num_count ++;
        }
    }
        
    sort_double(r_ext, num_count, sizeof(r_ext[0]));
    
    for(i =0; i< num_count-1 ; i++){
    
        if( r_ext[i] == r_ext[i+1] ){
        
            num_count --;
            for( j=i; j<num_count; j++){
            
                r_ext[j] = r_ext[j+1];
            }
        }
    }

    *r_num = num_count;

    
    return 0;
}


///// this function calculate the intersection's extreme distance from the center of the star
///// intersection is an overlap area of the circle and the ellipse
///// we have 10 candidates at most and these exetreme values are the upper or lower limits of radial integration of the intersection
void set_r_ext_circle_ellipse(double *ppos, double flat, double rp_ec, double rp_circ,  double *r_ext_set, int *r_set_count){
    
    
    double dist = sqrt (pow(ppos[0],2.0) + pow(ppos[1],2.0));
    double thetap_ext[2];
    int i,j;
    
    /// A is a x-value of the intersection of the circle and the ellipse measured from the center of the circle.
    double A = sqrt( (pow(rp_circ,2.0) - pow(rp_ec*(1-flat),2.0) )/(2*flat - pow(flat,2.0)));
    int flag[10]={0};
    
    /// we will insert 10 candidates into x_A[10]
    double x_A[10];
    
    ///1: we have 4 candidates considering the intersection of the ellipse and circle
    x_A[0] = A;x_A[1] = A; x_A[2] = -A;x_A[3] = -A;
    double y_A[10];
    y_A[0] = sqrt ( pow(rp_circ,2.0) - A*A);y_A[1] = -sqrt ( pow(rp_circ,2.0) - A*A);
    y_A[2] = sqrt ( pow(rp_circ,2.0) - A*A);y_A[3] = -sqrt ( pow(rp_circ,2.0) - A*A);
    
    
    
    
    // intersection of the planet and the line between the center of the star and the center of the planet
    x_A[4] = -ppos[0]*rp_circ/dist;
    y_A[4] = -ppos[1]*rp_circ/dist;
    
    x_A[5] = ppos[0]*rp_circ/dist;
    y_A[5] = ppos[1]*rp_circ/dist;
    
    if( pow(x_A[4] ,2.0)  + pow(y_A[4]/(1-flat),2.0)  > pow(rp_ec,2.0))
        flag[4] = 1;
    if( pow(x_A[5] ,2.0)  + pow(y_A[5]/(1-flat),2.0)  > pow(rp_ec,2.0))
        flag[5] = 1;
    

    
    
    /// we search points on the ellipse for a candidate
    double x_ext[4];
    double y_ext[4];
    double r_ext[4];
    int r_num_ellipse;

    /// obtain the extreme value for the eclipse
    solve_fourth_rext( ppos[0], ppos[1], rp_ec, flat, r_ext, &r_num_ellipse, x_ext, y_ext);
  //  printf("now\n");
    //printf("%f %f %f %f\n ~~~~~~~\n",r_ext[0],r_ext[1],r_ext[2],r_ext[3]);
    for ( i=0; i<4 ; i++){
   // printf("%f %f\n ~~~~\n",x_ext[i],y_ext[i]);
    }

    for ( i=0 ; i<r_num_ellipse; i++){
        x_A[i+6] = x_ext[i];
        y_A[i+6] = y_ext[i];
        if( pow(x_A[i+6],2.0) + pow(y_A[i+6],2.0) > pow(rp_circ,2.0))
            flag[i+6] = 1;
    }
    for ( i=r_num_ellipse ; i<4; i++){
        flag[i+6] = 1;
    }
    
    double r_min = 1e20;
    double r_max = 0.0;
    
    int r_count=0;
 
    
    for ( i=0; i< 10; i++){
    //printf("%d %f %f %f\n",i,x_A[i],y_A[i],sqrt ( pow(x_A[i] + ppos[0],2.0) + pow( y_A[i] + ppos[1],2.0) ));
        if (flag[i] != 0 ) continue;
   double r_now = sqrt ( pow(x_A[i] + ppos[0],2.0) + pow( y_A[i] + ppos[1],2.0) );
       r_ext_set[r_count++] = r_now;
    
    //printf("%d %f %f %f %f\n",i,x_A[i],y_A[i],r_min,r_max);
    }
 
    sort_double(r_ext_set,r_count,sizeof(r_ext[0]));
    
    for(i =0; i< r_count-1 ; i++){
    
        if( r_ext_set[i] == r_ext_set[i+1] ){
        
            r_count --;
            for( j=i; j<r_count; j++){
            
                r_ext_set[j] = r_ext_set[j+1];
            }
        }
    }

    *r_set_count = r_count;
    
    
    }



//// we derive intersections a "r" circle "" S "" centered at the origin and a surface of a product set of an elliptic disk( r_ec , r_ec (1-f)) "" E ""  and a disk (r_p) "" D "" centered at (ppos[0], ppos[1])
//// the results are given by the angle of the intersection measured from a reference line.
//// first we derive the each of the  intersection of (S and E) and ( S and D) and then we evaluate each point to confirm if
////  the results are
/////// we get four solutions at most and two solutions at least
void theta_ext_ellipse_circle(double *ppos, double r_ec, double r_p, double f, double r, double theta[4], int *num_solu){
    //printf("<theta_ext_ellipse_circle>\n");

    double z[8];
    double kai_x[4];
    double kai_y[4];
    int solu_count = 0;
    
    double x_now;
    double x_now_i;
    double y_can[2];
    int i;
    
    
    
    /// we derive the intersection of S and E
    solve_fourth_intersection(ppos, r_ec, r_ec * (1-f), r, z);
    
    
    
    /// we check the solutions and add them to the solution list if they are included in D
     for (i =0; i <4; i++){
         
            x_now = z[2*i] +ppos[0];
            x_now_i = z[2*i+1];
            y_can[0] =sqrt (fabs(pow(r,2.0) - pow(x_now,2.0)) ) ;
            y_can[1] = -sqrt (fabs( pow(r,2.0) - pow(x_now,2.0)) ) ;
            // y_can[0] = ppos[1] + (1-f) * sqrt( fabs(pow(r_ec,2.0) - pow(z[2*i],2.0)));
            //y_can[1] = ppos[1] -(1-f) * sqrt( fabs(pow(r_ec,2.0) - pow(z[2*i],2.0)));
            if( fabs(x_now_i/x_now) >1e-6) continue;

  
            // if we have multiple roots, we should deal with it carefully
            if(fabs(z[2*i] - z[2*i+2]) < fabs(z[2*i]) * 1e-8 && fabs(z[2*i+3]/z[2*i+2]) <1e-6 ){
            y_can[0] = ppos[1] + (1-f) * sqrt( fabs(pow(r_ec,2.0) - pow(z[2*i],2.0)));
            y_can[1] = ppos[1] -(1-f) * sqrt( fabs(pow(r_ec,2.0) - pow(z[2*i],2.0)));

                
                if( - pow(r,2.0) + pow(x_now,2.0) > 0 ){ i+=1 ;continue;}
                if( circle(ppos,x_now,y_can[0],r_p) >0 ){
                    kai_x[solu_count] = x_now;
                    kai_y[solu_count] = y_can[0];
                    solu_count++;
                   // printf("ok\n");
                }
               //     printf("ff22f%f %f\n",x_now, y_can[0]);
                if( circle(ppos,x_now,y_can[1],r_p) >0 ){
                    kai_x[solu_count] = x_now;
                    kai_y[solu_count] = y_can[1];
                    solu_count++;
                   // printf("ok\n");

                }
             //   printf("ff22f%f %f\n",x_now, y_can[1]);
                    i++;
            }
         
         
            /// if we have no mutiple roots we investigate the solution
            else{
                
                if( - pow(r,2.0) + pow(x_now,2.0) > 0) continue;

                if(fabs(daen(ppos,x_now, y_can[0],r_ec, r_ec*(1-f))) < fabs(daen(ppos,x_now, y_can[1],r_ec, r_ec*(1-f))) ){
                
                
                    if( circle(ppos,x_now,y_can[0],r_p) >0 ){
                        kai_x[solu_count] = x_now;
                        kai_y[solu_count] = y_can[0];
                        solu_count++;
                    }
                }
                else{
                    if( circle(ppos,x_now,y_can[1],r_p) >0 ){
                        kai_x[solu_count] = x_now;
                        kai_y[solu_count] = y_can[1];
                        solu_count++;
                    }
                }
                
            }
    }
    
    
    
    /// we derive the intersection of S and D
    /// and  we check the solutions and add them to the solution list if they are included in E
    if( r - r_p < sqrt(ppos[0] * ppos[0] + ppos[1] * ppos[1]) &&  sqrt(ppos[0] * ppos[0] + ppos[1] * ppos[1]) < r + r_p ){
    
        double theta_rot = atan2(ppos[1],ppos[0]);
        double r_fabs = sqrt ( pow( ppos[0],2.0) + pow(ppos[1] ,2.0) ) ;
        double x_circle_circle  = (pow(r,2.0) +pow(r_fabs,2.0)- pow(r_p,2.0))/(2*r_fabs);
        double y_circle_circle = sqrt( pow(r ,2.0) - pow(x_circle_circle,2.0));
        
        
        for ( i=0;i<2;i++){
        
            double x_circle = cos(theta_rot) * x_circle_circle - (1-2*i)* sin(theta_rot) * y_circle_circle;
            double y_circle = +sin(theta_rot) * x_circle_circle +(1-2*i) *cos(theta_rot)  *y_circle_circle;
            
            if( daen(ppos,x_circle, y_circle,r_ec, r_ec*(1-f)) >0){
                kai_x[solu_count] = x_circle;
                kai_y[solu_count] = y_circle;
                solu_count++;
                
            }
        }
    }
    
    
    //// Finally we get the solution
    for (i =0 ; i< solu_count ; i++){
        theta[i] = atan2(kai_y[i],kai_x[i]);
        if(theta[i] <0) theta[i] += 2 * M_PI;
    }
    
    sort_double(theta,solu_count,sizeof(theta[0]));
    *num_solu = solu_count;
}


/// we derive the intersection of an ellipse ( r_ec, f) centered at (ppos[0], ppo[1])  and a circle (r) centered at (0,0)
void theta_ext_ellipse(double *ppos, double r_ec, double f, double r, double theta[4], int *num_solu){

    double z[10];
    z[9] = 1e20;
    z[8] = 1e20;
    double kai_x[4];
    double kai_y[4];
    int solu_count = 0;
    
    double x_now;
    double x_now_i;
    double y_can[2];
    int i;
    
    solve_fourth_intersection(ppos, r_ec, r_ec * (1-f), r, z);
    for (i =0; i <4; i++){
            if( fabs(z[2*i]) > r_ec)continue;
            x_now = z[2*i] +ppos[0];
            x_now_i = z[2*i+1];
            y_can[0] =sqrt (fabs(pow(r,2.0) - pow(x_now,2.0)) ) ;
            y_can[1] = -sqrt (fabs( pow(r,2.0) - pow(x_now,2.0)) ) ;
            // y_can[0] = ppos[1] + (1-f) * sqrt( fabs(pow(r_ec,2.0) - pow(z[2*i],2.0)));
            //y_can[1] = ppos[1] -(1-f) * sqrt( fabs(pow(r_ec,2.0) - pow(z[2*i],2.0)));
            if( fabs(x_now_i/x_now) >1e-6) continue;
        
        
            /// if we have the same solution, we should deal with it manually
            if(fabs(z[2*i] - z[2*i+2]) < fabs(z[2*i]) * 1e-8 && fabs(z[2*i+3]/z[2*i+2]) <1e-6  ){

            y_can[0] = ppos[1] + (1-f) * sqrt( fabs(pow(r_ec,2.0) - pow(z[2*i],2.0)));
            y_can[1] = ppos[1] -(1-f) * sqrt( fabs(pow(r_ec,2.0) - pow(z[2*i],2.0)));


                if( - pow(r,2.0) + pow(x_now,2.0) > 0 ){ i+=1 ;continue;}

                    kai_x[solu_count] = x_now;
                    kai_y[solu_count] = y_can[0];
                    solu_count++;

                    kai_x[solu_count] = x_now;
                    kai_y[solu_count] = y_can[1];
                    solu_count++;
                    i++;
            }
            /// no multiple root at "i"
            else{
                
                if( - pow(r,2.0) + pow(x_now,2.0) > 0) continue;

                if(fabs(daen(ppos,x_now, y_can[0],r_ec, r_ec*(1-f))) < fabs(daen(ppos,x_now, y_can[1],r_ec, r_ec*(1-f))) ){
                    kai_x[solu_count] = x_now;
                    kai_y[solu_count] = y_can[0];
                    solu_count++;
                }
                else{
                    kai_x[solu_count] = x_now;
                    kai_y[solu_count] = y_can[1];
                        solu_count++;
                }
                
            }
    }

    
    for (i =0 ; i< solu_count ; i++){
    theta[i] = atan2(kai_y[i],kai_x[i]);
    if(theta[i] <0) theta[i] += 2 * M_PI;
    }
    

    sort_double(theta,solu_count,sizeof(theta[0]));
    *num_solu = solu_count;
    
}


//// get the position of the intersecton of the circle and the circle.
/// the results are given as a form of theta[2]
double theta_set_circle( double *pos, double rp, double r, double *theta, int *num_theta){
    double theta_pos = atan2(pos[1], pos[0]);
    if ( theta_pos < 0 ) theta_pos += 2 * M_PI;
    
    
    double dist_rp = sqrt( pow(pos[0],2.0) + pow(pos[1],2.0));
    
    if( dist_rp ==0 ){
        if ( rp < r){
            *num_theta = 0;
            return 0;
        }else{
            *num_theta = 2;
            theta[0] = 0;
            theta[1] = 2 * M_PI;
            return 0;
        }
    }
    
    
    if( dist_rp < rp){
        *num_theta =2;
        theta[0] = 0;
        theta[1] = 2 * M_PI;
    }
    if( dist_rp < r -rp || r+ rp < dist_rp){
        *num_theta =0;
        return 0;
    }
    if( dist_rp == r-rp){
        *num_theta =1;
        theta[0] = theta_pos;
        return 0;
    }
    if( dist_rp == r +rp){
        *num_theta =1;
        theta[0] = theta_pos;
        return 0;
    }
    
    double x =  ( pow(dist_rp,2.0) - pow(rp,2.0) + pow(r, 2.0))/(2 * dist_rp);
    double y = sqrt ( pow(r,2.0) - pow( x,2.0));
    double theta_circle = atan2(fabs(y), fabs( x));
    theta[0] = theta_pos - theta_circle;
    theta[1] = theta_pos + theta_circle;
    *num_theta =2;
    return 0;
}
    
#endif


