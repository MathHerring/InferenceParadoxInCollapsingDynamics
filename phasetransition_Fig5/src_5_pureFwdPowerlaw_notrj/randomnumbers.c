// subversion header
// $URL: svn+sf://jonnahs@svn.code.sf.net/p/iphigenie/svn/trunk/program/src/randomnumbers.c $
// $Rev: 2353 $
// $Author: g-mathias $
// $Date: 2014-10-07 18:07:15 +0200 (Di, 07 Okt 2014) $
//    This file is part of the MD package IPHIGENIE.
//    Copyright (C) 2014  "The iphigenie development team":
//    See the file PEOPLE that comes with this distribution.
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
// --------------------------------------------------------
/*
*  C Implementation: randomnumbers
* Author: Gerald Mathias <gerald.mathias@physik.uni-muenchen.de>, (C) 2012
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include "types.h"
//#include "iffi.h"
//#include "enumtypes.h"
#include "randomnumbers.h"
//#include "types.h"


/***********************************************************************/
/*** a very good random generator from the numerical recipes ***********/
/***********************************************************************/
#define IM1 2147483563 
/* IM1 = IA1*IQ1+IR1 */
#define IA1 40014
#define IQ1 53668
#define IR1 12211

#define IM2 2147483399
/* IM2 = IA2*IQ2+IR2 */
#define IA2 40692
#define IQ2 52774
#define IR2 3791
#define BASESEED -13L

#define RANPREC double 
#define EPS (1.0/(IM1-1))

#define RNMX (1.0-EPS)

#define NDIV (1+(IM1-1)/NTAB)



static STRUCT_RANDOM2_STATE random2State[NUM_SEEDS];

static inline long linearCongruentialSchrage(long int s,long int m, long int q, long int a, long int r){
    // calculate s = a*s % m by Schrages method to avoid a long overflow
    // with m = a*q + r 
   long int k = s/q;
   s = a*( s - k*q ) - k*r;
   if(s<0l) s+=m;
   return s;
}

static void initRandom2State(STRUCT_RANDOM2_STATE *rState){
    
    rState->seed[0] = max(labs(rState->seed[0]),1l);
    rState->seed[1] = rState->seed[0];
   
    for (int j = NTAB+7;  j>=0; j--) {
        /* update the first seed  */
        rState->seed[0]=linearCongruentialSchrage(rState->seed[0],IM1,IQ1,IA1,IR1);
        /* load state vector v after 7 warm ups */
        if (j<NTAB) rState->state[j] = rState->seed[0];
     }
     /* initialize result variable */
     rState->y = rState->state[0];
     rState->count = 0l;
}

STRUCT_RANDOM2_STATE* getRandom2Pointer(void){
    return random2State; 
}

// for restart
void setRandom2State(const STRUCT_RANDOM2_STATE rState, int entry){
    random2State[entry]=rState;
}

STRUCT_RANDOM2_STATE getRandom2State(int entry){
    return random2State[entry];    
}

//initRandSeedStructs(global_data->paraInfo.myPartitionId);
void initRandSeedStructs(int offset) {
    int entry;
    for( entry=0; entry<NUM_SEEDS; entry++){
         random2State[entry].seed[0]=  BASESEED - entry - NUM_SEEDS*offset;
         initRandom2State(&(random2State[entry]));
    }
}

static long int random2Generator(STRUCT_RANDOM2_STATE *rState){
    
    /* update the first seed  */
    rState->seed[0]=linearCongruentialSchrage(rState->seed[0],IM1,IQ1,IA1,IR1);
    /* update the second seed */
    rState->seed[1]=linearCongruentialSchrage(rState->seed[1],IM2,IQ2,IA2,IR2);

    /* choose state variable depending on last random number y */
    int j = rState->y / NDIV;
    rState->y = rState->state[j]-rState->seed[1];
    /* replace state variable with other seed */
    rState->state[j] = rState->seed[0];
    
    /* wrap output to positive numbers */
    if (rState->y < 1l) rState->y += IM1-1l;
    /* count how often we have been called */
    rState->count++;
    return rState->y;
}

double ranStandard(SEEDTYPE entry) {
     long int result = random2Generator(random2State+entry);
     return (RANPREC) min( RNMX,result *1.0/IM1);
}

int ranInt(int entry, int p){
    // largest result that still fits a multiple of p in IM1
    const long int maxRes = IM1 - IM1%p -2;
    long int res;
    do{
        res = random2Generator(random2State+entry);  // returns values 1,...,IM1-1
    }while(res >maxRes); // discard results near IM1 to preserve an equal distribution
    return (res-1)%p;
}

/* generate random numbers from normal distribution *
 * using the box mueller algorithm                      */
double gasdev(void) {
    static bool isSet=false;
    static double rndStore;

    if (!isSet) {
        double fac,r2,x,y;
        /* find random vector within a unit circle */
        do {
            x = 2.0*ranStandard(MISC_SEED)-1.0;
            y = 2.0*ranStandard(MISC_SEED)-1.0;
            r2=x*x +y*y ;
        } while (!(r2 < 1.0 && r2 > 0.0));
        
        fac=sqrt(-2.0*log(r2)/r2);
        /* store first random number for the next call */
        rndStore = x*fac;
        isSet    = true;
        /* return second random number */
        return y*fac;
    } else {
        /* return first random number from the last call */
        isSet = false;
        return rndStore;
    }
}

double gamdev(const int ia) {
    double x=0.0;
    //pexIf(ia < 1,"illegal call to gamdev\n");
    if (ia < 6) {
        x=1.0;
        for (int j=0; j<ia; j++) x *= ranStandard(MISC_SEED);
        x = -log(x);
    } else {
        double e;
        const double am = ia-1.0;
        const double s  = sqrt(2.0*am+1.0);
        do {
            double y;
            do {
                double v1,v2;
                do { /* random vector in [0,1]x[-1,1] */
                    v1 =   ranStandard(MISC_SEED);
                    v2 = 2.0*ranStandard(MISC_SEED)-1.0;
                } while (v1*v1+v2*v2 > 1.0); // of length <= 1.0
                y=v2/v1;  // tan of the angle
                x=s*y+am;
            } while (!(x > 0.0));
            e=(1.0+y*y)*exp(am*log(x/am)-s*y);
        } while (ranStandard(MISC_SEED) > e);
    }
    return x;
}

void getRandIndex(int *counts){

    for(int  entry=0; entry<NUM_SEEDS; entry++ ) {
        counts[entry] = random2State[entry].count;
    }
   return;
}
