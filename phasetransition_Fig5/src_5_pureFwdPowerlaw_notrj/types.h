#ifndef __TYPES_H__
#define __TYPES_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>



#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b)) 

typedef struct {
	double   D;
	double   alpha;
	double	 beta;
	double   gamma;

	int      nrealiz;        
	int      seed;
	double   dt;
	double   x_start_sim;
	double   var_start_sim;
	double   x_start_mean;
	double   x0_for_reverse;
	double   x_upper_bound;
	int      total_time;
	int      max_rtime; 

	int      wfreq;
	int      traFreq;
	int      N;
					
	double*  x;
	int      t_hit_max;
	         
	double*  xmean;
	double*  x2mean;
	double*  x3mean;
	int*     Norm;
	double** cov;
	int      t_rev_max;
	int      t_rev_max_wfreq;  
} CE;


#endif
