#ifndef SORT_H_
#define SORT_H_
#include <stdlib.h>

int compare_doubles (const void * a,
                 const void * b)
{
    if (*(double *)a > *(double *)b)
       return 1;
    else if (*(double *)a < *(double *)b)
       return -1;
    else
       return 0;
}

void sort_double(void *data, int num_data, int num_data_size){
    qsort(data, num_data, num_data_size, compare_doubles);}

#endif
