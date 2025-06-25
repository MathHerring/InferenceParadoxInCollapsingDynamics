#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "types.h"
#include "init.h"
#include "randomnumbers.h"

#define M_PI 3.14159265358979323846
#define PRINT_TRJ 1
#define PRINT_TRJ_FWD 0
#define PRINT_HITTING_TIME 1
#define UPPER_BOUND 1
#define READ_XMEAN 0
#define FREE_DIFFUSION 0
#define PURE_FWD_POWERLAW 1

CE* CE_create() {
	CE* tmp = (CE*) calloc((size_t)1,sizeof(CE));
	return tmp;
}

void CE_destroy(CE* ce) {
    if (ce != NULL) free(ce);
}

static void run(CE *ce){  
	int t_hit = 0;
	double countNonTerm = 0.0;
	double xmean2power;
	double var;
	double std;
	double* x_mean_read;

	FILE *fileReadXmean;
	FILE *fileMean;
	FILE *fileVar;
	FILE *fileCov;
	FILE *fileTrj;
	FILE *fileTrj_fwd;
	FILE *fileSkew;
	FILE *fileHittingTime;
	
	fileMean = fopen("num_mean.txt", "w+");	
	fileVar  = fopen("num_var.txt" , "w+");
	fileCov  = fopen("num_cov.txt" , "w+");
	fileSkew = fopen("num_skew.txt", "w+");

	if(PRINT_TRJ)fileTrj = fopen("num_trj.txt", "w+");
	if(PRINT_TRJ_FWD)fileTrj_fwd = fopen("num_trj_fwd.txt", "w+");
	if(PRINT_HITTING_TIME) fileHittingTime = fopen("hittingTime.txt", "w+");

	
	if(READ_XMEAN){
	  x_mean_read  = (double  *) calloc( ce->nrealiz,sizeof(double));
	  fileReadXmean  = fopen("start_values_qstat.txt" , "r+");
	  for(int j=0; j< ce->nrealiz; j++){
	    fscanf(fileReadXmean,"%lf\n",&(x_mean_read[j]) );
	  }
	}
	
	// main loop
	for(int j=0; j < ce->nrealiz; j++){
	    if((j%100)==0) printf("ensemble idx: %i\n",j);
	    if ((countNonTerm/ce->nrealiz) >= 0.9) {
			printf("\n");
			printf("Termiated as not collapsing\n");
			break;
	    }
	    for(int i=0; i < (ce->N-1); i++){
			// init distribution
			if (i== 0){
				ce->x = (double *) calloc(ce->N,sizeof(double));
				if(READ_XMEAN){
					if(x_mean_read[j] <= 0.0){
						printf("Exiting the program....\n");
						printf("....invalid or to view init coords\n");			
						exit(0);
					}
					ce->x_start_mean = x_mean_read[j];
					ce->x[i] = x_mean_read[j];
				}else{
					ce->x_start_mean = 0.0;
					while(ce->x_start_mean <= 0.0){
						ce->x_start_mean = ( (double)rand() / (double)((unsigned)RAND_MAX + 1) )*ce->var_start_sim + ce->x_start_sim;
					}
					ce->x[i] = ce->x_start_mean;
					if(j==0) printf("select starting position: %f\n",ce->x[0]);
				}
			}
			// boundaries
			// upper
			if (UPPER_BOUND && ce->x[i] >= ce->x_upper_bound)ce->x[i] = ce->x_upper_bound - fabs(ce->x[i] - ce->x_upper_bound);
			// lower
			if (ce->x[i]<= ce->x0_for_reverse){		  
						// absorbing    
				ce->x[i]= ce->x0_for_reverse;
				t_hit = i;          
				if (t_hit > ce->t_hit_max) ce->t_hit_max = t_hit;   
				break;
			}

			// Equations of motion ---------------------------------------------------------
			if(FREE_DIFFUSION){
				if((i==0) &&(j==0) ) printf("FREE_DIFFUSION\n");
				ce->x[i+1] = ce->x[i] + sqrt(ce->D *ce->dt)*gasdev(); 
			}else if(PURE_FWD_POWERLAW){
				if((i==0) &&(j==0) ) printf("PURE_FWD_POWERLAW\n");
				ce->x[i+1] = ce->x[i] 
					- ce->gamma * pow(ce->x[i],ce->alpha)*ce->dt
					+ sqrt(ce->D*pow(ce->x[i],ce->beta) *ce->dt)*gasdev(); 		    
			}else{
				if((i==0) &&(j==0) ) printf("NO FUNKTION\n");
			}
			// ---------------------------------------------------------
			// max N reached	
			if(i == ce->N - 2){ 
			t_hit = ce->N - 2;
			}	
	    }

	    // print hitting total_time
	    if(PRINT_HITTING_TIME){
			int int_t_hit = (int)t_hit;
			if(int_t_hit == (ce->N-2)){
				fprintf(fileHittingTime,"\n");
	      	}else{
				fprintf(fileHittingTime,"%14.6f\n",t_hit*ce->dt);
	      	}
	    }
	    
	    // revert and average
	    int int_t_hit = (int)t_hit;
	    if(int_t_hit == (ce->N-2)){
			countNonTerm += 1.0;
			printf("nonTerm");
			printf("\n");
	    } else{
			int t_rev = ((int)(ce->t_rev_max) < int_t_hit) ? ce->t_rev_max : int_t_hit;		
			// --1-- covariance
			for(int ii = 0.0; ii <= t_rev; ii+= ce->wfreq){
				for(int jj = 0.0; jj <= t_rev; jj+= ce->wfreq){
					if(!(ce->Norm[ii] == 0.0 || ce->Norm[jj] == 0.0)){
						int norm = min(ce->Norm[ii],ce->Norm[jj]);
						int iidx = (int)(ii/ce->wfreq);
						int jidx = (int)(jj/ce->wfreq);
						double XX = ( ce->x[int_t_hit-ii] - (ce->xmean[ii] / ce->Norm[ii]) );
						double YY = ( ce->x[int_t_hit-jj] - (ce->xmean[jj] / ce->Norm[jj]) );
						ce->cov[iidx][jidx]      += XX*YY*(norm) /(norm+1.0); 
					}
				}
			}
		
			// print forward trajectory
			if(PRINT_TRJ_FWD){
				int tag_first_zero = 0;
				for(int ii=0; ii<= (int)(ce->N/ce->traFreq); ii++){
					if(ce->x[ii*ce->traFreq] > 0.0) fprintf(fileTrj_fwd,"%14.6f",ce->x[ii*ce->traFreq]);
						else if(tag_first_zero == 0 ){ 
						fprintf(fileTrj_fwd,"%14.6f", 0.0);
						tag_first_zero = 1;
					}
					else fprintf(fileTrj_fwd,"%14.6f", 0.0/0);
				}
				fprintf(fileTrj_fwd,"\n");
			}				
		
			if(PRINT_TRJ){
				for(int ii=0; ii<= (int)(ce->t_rev_max/ce->traFreq); ii++){
					if( (int)(ii*ce->traFreq)<= t_rev ) fprintf(fileTrj,"%14.6f",ce->x[int_t_hit-ii*ce->traFreq]);
					else fprintf(fileTrj,"%14.6f", 0.0/0);		    
				}
			}
			
					
			// --2-- mean and variance
			for(int ii = 0.0; ii <= t_rev; ii++){
				ce->xmean [ii]    += ce->x[int_t_hit-ii];
				ce->x2mean[ii]    += ce->x[int_t_hit-ii]*ce->x[int_t_hit-ii];
				ce->x3mean[ii]    += ce->x[int_t_hit-ii]*ce->x[int_t_hit-ii]*ce->x[int_t_hit-ii];
				ce->Norm  [ii]    += 1.0;
			}
	    }
	    if(PRINT_TRJ) fprintf(fileTrj,"\n");
	    free(ce->x);
	}
          
	// Normalization of mean and variance    
	for(int ii=0; ii< ce->t_rev_max; ii++){
		if(ce->Norm[ii]  == 0.0){
			ce->xmean[ii]   = 0.0;
			ce->x2mean[ii]  = 0.0;
			ce->x3mean[ii]  = 0.0;
		} else{
			ce->xmean[ii]  /= ce->Norm[ii];
			ce->x2mean[ii] /= ce->Norm[ii];
			ce->x3mean[ii] /= ce->Norm[ii];
			
			xmean2power = ce->xmean[ii]*ce->xmean[ii];
			var 	    = ce->x2mean[ii]-xmean2power;
			std 	    = sqrt(var);

			ce->x3mean[ii]  = (ce->x3mean[ii]-3.0*ce->x2mean[ii]*ce->xmean[ii] + 2.0*xmean2power*ce->xmean[ii]) / (std*std*std);    
			ce->x2mean[ii]  =  var;
		}
	}	
	  
	// Normalization of covariance
	for(int ii = 0.0; ii < ce->t_rev_max_wfreq; ii++){
		for(int jj = 0.0; jj < ce->t_rev_max_wfreq; jj++){
			int idx = ii * ce->wfreq;
			int jdx = jj * ce->wfreq;
			int norm = min(ce->Norm[idx],ce->Norm[jdx]);
			if(norm == 0.0){
				ce->cov[ii][jj] = 0.0;
			} else{
				ce->cov[ii][jj]/=norm;
			}
		}
	}
	
	// Output
	for(int ii=0; ii< ce->t_rev_max; ii++){
	  	fprintf(fileMean,"%f\n",ce->xmean[ii]);
	}		
	for(int ii=0; ii< ce->t_rev_max; ii++){
	 	 fprintf(fileVar,"%f\n",ce->x2mean[ii]);
	}
	for(int ii=0; ii< ce->t_rev_max; ii++){
	 	 fprintf(fileSkew,"%f\n",ce->x3mean[ii]);
	}	
	for(int ii = 0.0; ii < ce->t_rev_max_wfreq; ii++){
		for(int jj = 0.0; jj < ce->t_rev_max_wfreq; jj++){    
			fprintf(fileCov,"%f\n",ce->cov[ii][jj]);
		}
	}
	
	fclose(fileMean);
	fclose(fileVar);
	fclose(fileCov);
	fclose(fileSkew);
	if(READ_XMEAN)fclose(fileReadXmean);
	if(PRINT_TRJ )fclose(fileTrj);
	if(PRINT_TRJ_FWD)fclose(fileTrj_fwd);
	if(PRINT_HITTING_TIME)fclose(fileHittingTime);
}

int main(int argc, char** argv){
	CE* ce = CE_create();
      
	switch (argc) {
		default:
		case 16: ce->beta		 	= atof(argv[15]);
		case 15: ce->alpha	        = atof(argv[14]);
		case 14: ce->gamma   	 	= atof(argv[13]);	      
		case 13: ce->var_start_sim  = atof(argv[12]);
		case 12: ce->x_upper_bound  = atof(argv[11]);
		case 11: ce->max_rtime	 	= atoi(argv[10]);            
		case 10: ce->traFreq 	 	= atoi(argv[9 ]); 
		case 9 : ce->wfreq	 	 	= atoi(argv[8 ]);
		case 8 : ce->total_time 	= atoi(argv[7 ]);
		case 7 : ce->x0_for_reverse = atof(argv[6 ]);
		case 6 : ce->D   		 	= atof(argv[5 ]);
		case 5 : ce->dt 		 	= atof(argv[4 ]);
		case 4 : ce->seed 		 	= atoi(argv[3 ]);
		case 3 : ce->nrealiz 	 	= atoi(argv[2 ]);
		case 2 : ce->x_start_sim 	= atof(argv[1 ]);	    
		case 1 : break;
	}
	
	printf("# # # # P A R A M E T E R S # # # # # \n");
	printf("var_start_sim    %f\n",ce->var_start_sim     );
	printf("x_upper_bound    %f\n",ce->x_upper_bound     );
	printf("max_rtime        %i\n",ce->max_rtime         );
	printf("traFreq          %i\n",ce->traFreq 	         );
	printf("wfreq            %i\n",ce->wfreq	         );
	printf("total_time       %i\n",ce->total_time        );
	printf("x0_for_reverse   %f\n",ce->x0_for_reverse    );
	printf("D                %f\n",ce->D   	         	 );
	printf("dt               %f\n", ce->dt 	         	 );
	printf("seed             %i\n",ce->seed 	         );
	printf("nrealiz          %i\n",ce->nrealiz 	         );
	printf("x_start_sim      %f\n",ce->x_start_sim       );
	printf("============================================== \n");
	
	CE_init(ce);
	run(ce);
	            
	// free mean, var and cov    
	free(ce->xmean);	
	free(ce->x2mean);
	free(ce->x3mean);
	free(ce->Norm);
	
	for(int i=0; i<= ce->t_rev_max_wfreq; i++) free(ce->cov[i]);
	free(ce->cov);
	
	CE_destroy(ce);
}
