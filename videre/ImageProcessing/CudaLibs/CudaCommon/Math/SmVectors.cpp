/********************************************************************************************
                            Small Vector  Class                            Harry Direen
                                                                           Jan. 2, 2015

                           Class Header File and Definitions

Notes:
The vectors in this Class / Structure are designed to be small... 2 or 3 elements.
The vectors are established as fixed arrays in the structure so that memory alocations
are not required... for larger vectors and matricies use the matrix_x class.

The max vector size is established by a constant at the top of this file.  If larger
vectors are required... then this number must be increased and the code re-compiled.
By avoiding mallocs to allocate vectors, the code should run much faster and work
well on cuda cores.

********************************************************************************************/


#include "SmVectors.h"
#include <math.h>
#include <stdio.h>

#pragma warning(disable:4270)

#ifndef AbsVal
#define AbsVal(x) (x < 0 ? -x : x)
#endif

#ifndef ToIntRnd
#define ToIntRnd(x) int(x < 0 ? x - 0.5 : x + 0.5)
#endif


/***************************************/
// Matrix Class for type float

SmVec_i SmVec_f::mk_integer()  // create a copy which is of type int...
{
    SmVec_i outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = ToIntRnd(Vec[i]);

    return outV;
}

SmVec_d SmVec_f::mk_double()
{
    SmVec_d outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = (double) Vec[i];

    return outV;
}

SmVec_f SmVec_f::element_product(const SmVec_f &A)
{
    SmVec_f outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = Vec[i] * A.Vec[i];
    return outV;
}

//Note: this is an element by element divide.
SmVec_f SmVec_f::element_divide(const SmVec_f &A)
{
    SmVec_f outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = Vec[i] / A.Vec[i];
    return outV;
}


float SmVec_f::l2norm()
{
    float result = 0;
    for (int i = 0; i < _vecSize; i++)
        result += Vec[i] * Vec[i];
    return (float) sqrt(result);
}

//
float SmVec_f::l2norm_sqrd()
{
    float result = 0;
    for (int i = 0; i < _vecSize; i++)
        result += Vec[i] * Vec[i];
    return result;
}

float SmVec_f::l1norm()
{
    double result = 0;
    for (int i = 0; i < _vecSize; ++i)
        result += fabs(Vec[i]);
    return (float) result;
}

float SmVec_f::inf_norm()
{
    double result = 0;
    for (int i = 0; i < _vecSize; ++i)
        result = result < fabs(Vec[i]) ? fabs(Vec[i]) : result;
    return (float) result;
}

float SmVec_f::inner_prod(const SmVec_f &y)
{
    float result = 0;
    for (int i = 0; i < _vecSize; i++)
        result += Vec[i] * y.Vec[i];
    return result;
}

void SmVec_f::print(const std::string &mess)
{
    printf("%s", mess.c_str());
    for (int i = 0; i < _vecSize; i++)
        printf(" %g", Vec[i]);

    printf("\n");
}

SmVec_i SmVec_d::mk_integer()  // create a copy which is of type int...
{
    SmVec_i outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = ToIntRnd(Vec[i]);

    return outV;
}

SmVec_f SmVec_d::mk_float()
{
    SmVec_f outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = (float) Vec[i];

    return outV;
}

double SmVec_d::l2norm()
{
    double result = 0;
    for (int i = 0; i < _vecSize; i++)
        result += Vec[i] * Vec[i];
    return sqrt(result);
}

//
double SmVec_d::l2norm_sqrd()
{
    double result = 0;
    for (int i = 0; i < _vecSize; i++)
        result += Vec[i] * Vec[i];
    return result;
}

double SmVec_d::l1norm()
{
    double result = 0;
    for (int i = 0; i < _vecSize; ++i)
        result += fabs(Vec[i]);
    return (double) result;
}

//
double SmVec_d::inf_norm()
{
    double result = 0;
    for (int i = 0; i < _vecSize; ++i)
        result = result < fabs(Vec[i]) ? fabs(Vec[i]) : result;
    return result;
}

double SmVec_d::inner_prod(const SmVec_d &y)
{
    double result = 0;
    for (int i = 0; i < _vecSize; i++)
        result += Vec[i] * y.Vec[i];
    return result;
}

void SmVec_d::print(const std::string &mess)
{
    printf("%s", mess.c_str());
    for (int i = 0; i < _vecSize; i++)
        printf(" %g", Vec[i]);

    printf("\n");
}


SmVec_d SmVec_i::mk_double()  // create a copy which is of type int...
{
    SmVec_d outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = (double) Vec[i];

    return outV;
}

SmVec_f SmVec_i::mk_float()  // create a copy which is of type integer...
{
    SmVec_f outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = (float) Vec[i];

    return outV;
}

double SmVec_i::l2norm()
{
    double result = 0;
    for (int i = 0; i < _vecSize; i++)
        result += Vec[i] * Vec[i];
    return sqrt(result);
}

//
double SmVec_i::l2norm_sqrd()
{
    double result = 0;
    for (int i = 0; i < _vecSize; i++)
        result += Vec[i] * Vec[i];
    return result;
}

double SmVec_i::l1norm()
{
    double result = 0;
    for (int i = 0; i < _vecSize; ++i)
        result += AbsVal(Vec[i]);
    return (double) result;
}

//
double SmVec_i::inf_norm()
{
    double result = 0;
    for (int i = 0; i < _vecSize; ++i)
        result = result < AbsVal(Vec[i]) ? AbsVal(Vec[i]) : result;
    return result;
}

double SmVec_i::inner_prod(const SmVec_i &y)
{
    double result = 0;
    for (int i = 0; i < _vecSize; i++)
        result += Vec[i] * y.Vec[i];
    return result;
}

void SmVec_i::print(const std::string &mess)
{
    printf("%s", mess.c_str());
    for (int i = 0; i < _vecSize; i++)
        printf(" %d", Vec[i]);

    printf("\n");
}
