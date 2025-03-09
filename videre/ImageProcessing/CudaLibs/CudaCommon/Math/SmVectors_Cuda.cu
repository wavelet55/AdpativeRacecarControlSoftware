/********************************************************************************************
                            Small Vector  Class                            Harry Direen
                                                                           Jan. 3, 2015

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

#include "SmVectors_Cuda.h"


__device__
SmVecCuda_i SmVecCuda_f::mk_integer()
{
    SmVecCuda_i outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = ToIntRnd(Vec[i]);

    return outV;
}

__device__
SmVecCuda_d SmVecCuda_f::mk_double()
{
    SmVecCuda_d outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = (double) Vec[i];

    return outV;
}


__device__
SmVecCuda_i SmVecCuda_d::mk_integer()
{
    SmVecCuda_i outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = ToIntRnd(Vec[i]);

    return outV;
}

__device__
SmVecCuda_f SmVecCuda_d::mk_float()
{
    SmVecCuda_f outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = (float) Vec[i];

    return outV;
}

__device__
SmVecCuda_d SmVecCuda_i::mk_double()
{
    SmVecCuda_d outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = (double) Vec[i];

    return outV;
}

__device__
SmVecCuda_f SmVecCuda_i::mk_float()
{
    SmVecCuda_f outV(_vecSize);
    for (int i = 0; i < _vecSize; i++)
        outV.Vec[i] = (float) Vec[i];

    return outV;
}
