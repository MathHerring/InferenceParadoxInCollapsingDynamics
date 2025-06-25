// subversion header
// $URL: svn+sf://jonnahs@svn.code.sf.net/p/iphigenie/svn/trunk/program/src/randomnumbers.h $
// $Rev: 2362 $
// $Author: g-mathias $
// $Date: 2014-10-14 16:47:27 +0200 (Di, 14 Okt 2014) $
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
//  C Interface: randomnumbers
// 
//  Description: random number generators and distributions form
//               numerical recipies 


#ifndef _RANDOMNUMBERS_H_
#define _RANDOMNUMBERS_H_
#define NTAB 32
typedef enum {CLU_SEED=0, LOAD_SEED, MISC_SEED,EXCHANGE_SEED, NUM_SEEDS} SEEDTYPE;
typedef struct{
    unsigned long count;        /* index of the random number */
    long seed[2]    ;  /* idum, idum2 */
    long y       ;     /* the random integer to be returned */
    long state[NTAB] ; /* state vector */
} STRUCT_RANDOM2_STATE;

typedef struct{
    long myseed;
    long rancount;
    long idum2;
    long iy;
    long iv[NTAB];
} STRUCT_SEED;

void   initRandSeedStructs(int offset);
double ranStandard(SEEDTYPE entry) ;
double gasdev(void) ;
double gamdev(const int ia);
void   getRandIndex(int *counts);

void setRandom2State(const STRUCT_RANDOM2_STATE rState, int entry);
STRUCT_RANDOM2_STATE getRandom2State(int entry);
STRUCT_RANDOM2_STATE* getRandom2Pointer(void);
int ranInt(int entry, int p);

#endif
