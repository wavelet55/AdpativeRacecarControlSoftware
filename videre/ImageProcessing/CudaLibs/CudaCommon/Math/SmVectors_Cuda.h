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



#ifndef    SMVECTORCUDA_H_
#define    SMVECTORCUDA_H_

#include <cuda_runtime.h>
#include <math.h>
#include "SmVectors.h"

#pragma warning(disable:4270)

#ifndef AbsVal
#define AbsVal(x) (x < 0 ? -x : x)
#endif

#ifndef ToIntRnd
#define ToIntRnd(x) int(x < 0 ? x - 0.5 : x + 0.5)
#endif


class SmVecCuda_d;

class SmVecCuda_i;

//This defines the maximum vector size that can be used.
//All vectors will set-aside this many elements even
//if the vector is of smaller dimention.
const int MAX_VECTOR_SIZE_CUDA = 3;


/***************************************/
// Matrix Class for type float

class SmVecCuda_f
{

private:
    int _vecSize;

public:
    float Vec[MAX_VECTOR_SIZE_CUDA];

    __device__
    int VecSize()
    {
        return _vecSize;
    }

    __device__
    inline SmVecCuda_f(int nrows = 1)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE_CUDA ? MAX_VECTOR_SIZE_CUDA : nrows;
    }

    __device__
    inline SmVecCuda_f(int nrows, float ival)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE_CUDA ? MAX_VECTOR_SIZE_CUDA : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ival;
    }

    __device__
    inline SmVecCuda_f(int nrows, const float *ivals)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE_CUDA ? MAX_VECTOR_SIZE_CUDA : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ivals[i];
    }

    __device__
    inline SmVecCuda_f(const SmVecCuda_f &A)        // copy constructor
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];
    }

    __device__
    ~SmVecCuda_f()
    {}                // destructor


    __device__
    inline float &val(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    __device__
    inline float getVal(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    __device__
    inline SmVecCuda_f operator=(const SmVecCuda_f &A)
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];

        return *this;
    }


    //Make a copy of this...
    //Used for Compatiblity
    __device__
    inline SmVecCuda_f copy()
    {
        return *this;
    }

    __device__
    SmVecCuda_i mk_integer();

    __device__
    SmVecCuda_d mk_double();


    __device__
    inline SmVecCuda_f operator+(const SmVecCuda_f &A)
    {
        SmVecCuda_f sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] + A.Vec[i];

        return sum;
    }

    __device__
    inline void operator+=(const SmVecCuda_f &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    __device__
    inline SmVecCuda_f operator-(const SmVecCuda_f &A)
    {
        SmVecCuda_f sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] - A.Vec[i];

        return sum;
    }

    __device__
    inline void operator-=(const SmVecCuda_f &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    __device__
    inline SmVecCuda_f operator-()
    {
        SmVecCuda_f sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = -Vec[i];

        return sum;
    }

    //Note: this is an element by element product... not the
    //inner product of two vectors.
    __device__
    SmVecCuda_f operator*(const SmVecCuda_f &A)
    {
        SmVecCuda_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }

    //Note: this is an element by element divide.
    __device__
    SmVecCuda_f operator/(const SmVecCuda_f &A)
    {
        SmVecCuda_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }


    __device__
    inline SmVecCuda_f operator*(const float x)
    {
        SmVecCuda_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = x * Vec[i];
        return outV;
    }

    __device__
    inline void operator*=(const float x)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] *= x;
    }

    __device__
    inline friend SmVecCuda_f operator*(const float x, const SmVecCuda_f &A)
    {
        SmVecCuda_f outV(A._vecSize);
        for (int i = 0; i < A._vecSize; i++)
            outV.Vec[i] = x * A.Vec[i];
        return outV;
    }


    // The next few routines are relational... A < B iff A[i][j] < B[i][j] for all i & j.
    // The routines return a 1 if true a zero otherwise.
    //int	operator<(const SmVecCuda_f &B); 
    //int	operator<=(const SmVecCuda_f &B); 
    //int	operator>(const SmVecCuda_f &B); 
    //int	operator>=(const SmVecCuda_f &B); 
    //int	operator==(const SmVecCuda_f &B); 
    //int	operator!=(const SmVecCuda_f &B); 

    // The next few routines work on every element of the matrix using the 
    // same scaler value:

    __device__
    inline SmVecCuda_f operator+(const float a)    // Adds "a" to every element of the matrix.
    {
        SmVecCuda_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] + a;
        return outV;
    }

    __device__
    inline void operator+=(const float a)   // Adds "a" to every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += a;
    }


    __device__
    inline SmVecCuda_f operator-(const float a)    // Subt "a" from every element of the matrix.
    {
        SmVecCuda_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] - a;
        return outV;
    }


    __device__
    inline void operator-=(const float a)   // Subt "a" from every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] -= a;
    }


    __device__
    inline void operator=(const float a)    // Sets every element of the matrix to "a".
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = a;
    }


    __device__
    SmVecCuda_f element_product(const SmVecCuda_f &A)
    {
        SmVecCuda_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }


    //Note: this is an element by element divide.
    __device__
    SmVecCuda_f element_divide(const SmVecCuda_f &A)
    {
        SmVecCuda_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }

    __device__
    float l2norm()
    {
        float result = 0;
        for (int i = 0; i < _vecSize; i++)
            result += Vec[i] * Vec[i];
        return (float) sqrt(result);
    }

    __device__
    float l2norm_sqrd()
    {
        float result = 0;
        for (int i = 0; i < _vecSize; i++)
            result += Vec[i] * Vec[i];
        return result;
    }

    __device__
    float l1norm()
    {
        float result = 0;
        for (int i = 0; i < _vecSize; ++i)
            result += fabs(Vec[i]);
        return result;
    }

    __device__
    float inf_norm()
    {
        float result = 0;
        for (int i = 0; i < _vecSize; ++i)
            result = result < fabs(Vec[i]) ? fabs(Vec[i]) : result;
        return result;
    }

    __device__
    float inner_prod(const SmVecCuda_f &y)
    {
        float result = 0;
        for (int i = 0; i < _vecSize; i++)
            result += Vec[i] * y.Vec[i];
        return result;
    }

};

class SmVecCuda_d
{

private:
    int _vecSize;

public:
    double Vec[MAX_VECTOR_SIZE_CUDA];

    __device__
    int VecSize()
    {
        return _vecSize;
    }

    __device__
    inline SmVecCuda_d(int nrows = 1)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE_CUDA ? MAX_VECTOR_SIZE_CUDA : nrows;
    }

    __device__
    inline SmVecCuda_d(int nrows, double ival)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE_CUDA ? MAX_VECTOR_SIZE_CUDA : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ival;
    }

    __device__
    inline SmVecCuda_d(int nrows, const double *ivals)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE_CUDA ? MAX_VECTOR_SIZE_CUDA : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ivals[i];
    }

    __device__
    inline SmVecCuda_d(const SmVecCuda_d &A)        // copy constructor
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];
    }

    __device__
    ~SmVecCuda_d()
    {}                // destructor


    __device__
    inline double &val(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    __device__
    inline double getVal(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    __device__
    inline SmVecCuda_d operator=(const SmVecCuda_d &A)
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];

        return *this;
    }


    //Make a copy of this...
    //Used for Compatiblity
    __device__
    inline SmVecCuda_d copy()
    {
        return *this;
    }

    __device__
    SmVecCuda_i mk_integer();

    __device__
    SmVecCuda_f mk_float();

    __device__
    inline SmVecCuda_d operator+(const SmVecCuda_d &A)
    {
        SmVecCuda_d sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] + A.Vec[i];

        return sum;
    }

    __device__
    inline void operator+=(const SmVecCuda_d &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    __device__
    inline SmVecCuda_d operator-(const SmVecCuda_d &A)
    {
        SmVecCuda_d sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] - A.Vec[i];

        return sum;
    }

    __device__
    inline void operator-=(const SmVecCuda_d &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    __device__
    inline SmVecCuda_d operator-()
    {
        SmVecCuda_d sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = -Vec[i];

        return sum;
    }

    //Note: this is an element by element product... not the
    //inner product of two vectors.
    __device__
    SmVecCuda_d operator*(const SmVecCuda_d &A)
    {
        SmVecCuda_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }

    //Note: this is an element by element divide.
    __device__
    SmVecCuda_d operator/(const SmVecCuda_d &A)
    {
        SmVecCuda_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }


    __device__
    inline SmVecCuda_d operator*(const double x)
    {
        SmVecCuda_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = x * Vec[i];
        return outV;
    }

    __device__
    inline void operator*=(const double x)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] *= x;
    }

    __device__
    inline friend SmVecCuda_d operator*(const double x, const SmVecCuda_d &A)
    {
        SmVecCuda_d outV(A._vecSize);
        for (int i = 0; i < A._vecSize; i++)
            outV.Vec[i] = x * A.Vec[i];
        return outV;
    }


    // The next few routines are relational... A < B iff A[i][j] < B[i][j] for all i & j.
    // The routines return a 1 if true a zero otherwise.
    //int	operator<(const SmVecCuda_d &B); 
    //int	operator<=(const SmVecCuda_d &B); 
    //int	operator>(const SmVecCuda_d &B); 
    //int	operator>=(const SmVecCuda_d &B); 
    //int	operator==(const SmVecCuda_d &B); 
    //int	operator!=(const SmVecCuda_d &B); 

    // The next few routines work on every element of the matrix using the 
    // same scaler value:

    __device__
    inline SmVecCuda_d operator+(const double a)    // Adds "a" to every element of the matrix.
    {
        SmVecCuda_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] + a;
        return outV;
    }

    __device__
    inline void operator+=(const double a)   // Adds "a" to every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += a;
    }


    __device__
    inline SmVecCuda_d operator-(const double a)    // Subt "a" from every element of the matrix.
    {
        SmVecCuda_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] - a;
        return outV;
    }


    __device__
    inline void operator-=(const double a)   // Subt "a" from every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] -= a;
    }


    __device__
    inline void operator=(const double a)    // Sets every element of the matrix to "a".
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = a;
    }


    __device__
    inline SmVecCuda_d element_product(const SmVecCuda_d &A)
    {
        SmVecCuda_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }

    //Note: this is an element by element divide.
    __device__
    inline SmVecCuda_d element_divide(const SmVecCuda_d &A)
    {
        SmVecCuda_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }


    __device__
    double l2norm()
    {
        double result = 0;
        for (int i = 0; i < _vecSize; i++)
            result += Vec[i] * Vec[i];
        return sqrt(result);
    }

    __device__
    double l2norm_sqrd()
    {
        double result = 0;
        for (int i = 0; i < _vecSize; i++)
            result += Vec[i] * Vec[i];
        return result;
    }

    __device__
    double l1norm()
    {
        double result = 0;
        for (int i = 0; i < _vecSize; ++i)
            result += fabs(Vec[i]);
        return (double) result;
    }
    //
    __device__
    double inf_norm()
    {
        double result = 0;
        for (int i = 0; i < _vecSize; ++i)
            result = result < fabs(Vec[i]) ? fabs(Vec[i]) : result;
        return result;
    }

    __device__
    double inner_prod(const SmVecCuda_d &y)
    {
        double result = 0;
        for (int i = 0; i < _vecSize; i++)
            result += Vec[i] * y.Vec[i];
        return result;
    }

};

class SmVecCuda_i
{

private:
    int _vecSize;

public:
    int Vec[MAX_VECTOR_SIZE_CUDA];

    __device__
    int VecSize()
    {
        return _vecSize;
    }

    __device__
    inline SmVecCuda_i(int nrows = 1)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE_CUDA ? MAX_VECTOR_SIZE_CUDA : nrows;
    }

    __device__
    inline SmVecCuda_i(int nrows, int ival)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE_CUDA ? MAX_VECTOR_SIZE_CUDA : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ival;
    }

    __device__
    inline SmVecCuda_i(int nrows, const int *ivals)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE_CUDA ? MAX_VECTOR_SIZE_CUDA : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ivals[i];
    }

    __device__
    inline SmVecCuda_i(const SmVecCuda_i &A)        // copy constructor
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];
    }

    __device__
    ~SmVecCuda_i()
    {}                // destructor


    __device__
    inline int &val(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    __device__
    inline int getVal(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    __device__
    inline SmVecCuda_i operator=(const SmVecCuda_i &A)
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];

        return *this;
    }


    //Make a copy of this...
    //Used for Compatiblity
    __device__
    inline SmVecCuda_i copy()
    {
        return *this;
    }

    __device__
    SmVecCuda_d mk_double();

    __device__
    SmVecCuda_f mk_float();

    __device__
    inline SmVecCuda_i operator+(const SmVecCuda_i &A)
    {
        SmVecCuda_i sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] + A.Vec[i];

        return sum;
    }

    __device__
    inline void operator+=(const SmVecCuda_i &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    __device__
    inline SmVecCuda_i operator-(const SmVecCuda_i &A)
    {
        SmVecCuda_i sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] - A.Vec[i];

        return sum;
    }

    __device__
    inline void operator-=(const SmVecCuda_i &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    __device__
    inline SmVecCuda_i operator-()
    {
        SmVecCuda_i sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = -Vec[i];

        return sum;
    }

    //Note: this is an element by element product... not the
    //inner product of two vectors.
    __device__
    SmVecCuda_i operator*(const SmVecCuda_i &A)
    {
        SmVecCuda_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }

    //Note: this is an element by element divide.
    __device__
    SmVecCuda_i operator/(const SmVecCuda_i &A)
    {
        SmVecCuda_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }


    __device__
    inline SmVecCuda_i operator*(const int x)
    {
        SmVecCuda_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = x * Vec[i];
        return outV;
    }

    __device__
    inline void operator*=(const int x)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] *= x;
    }

    __device__
    inline friend SmVecCuda_i operator*(const int x, const SmVecCuda_i &A)
    {
        SmVecCuda_i outV(A._vecSize);
        for (int i = 0; i < A._vecSize; i++)
            outV.Vec[i] = x * A.Vec[i];
        return outV;
    }


    // The next few routines are relational... A < B iff A[i][j] < B[i][j] for all i & j.
    // The routines return a 1 if true a zero otherwise.
    //int	operator<(const SmVecCuda_i &B); 
    //int	operator<=(const SmVecCuda_i &B); 
    //int	operator>(const SmVecCuda_i &B); 
    //int	operator>=(const SmVecCuda_i &B); 
    //int	operator==(const SmVecCuda_i &B); 
    //int	operator!=(const SmVecCuda_i &B); 

    // The next few routines work on every element of the matrix using the 
    // same scaler value:

    __device__
    inline SmVecCuda_i operator+(const int a)    // Adds "a" to every element of the matrix.
    {
        SmVecCuda_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] + a;
        return outV;
    }

    __device__
    inline void operator+=(const int a)   // Adds "a" to every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += a;
    }


    __device__
    inline SmVecCuda_i operator-(const int a)    // Subt "a" from every element of the matrix.
    {
        SmVecCuda_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] - a;
        return outV;
    }


    __device__
    inline void operator-=(const int a)   // Subt "a" from every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] -= a;
    }


    __device__
    inline void operator=(const int a)    // Sets every element of the matrix to "a".
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = a;
    }


    __device__
    inline SmVecCuda_i element_product(const SmVecCuda_i &A)
    {
        SmVecCuda_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }

    //Note: this is an element by element divide.
    __device__
    inline SmVecCuda_i element_divide(const SmVecCuda_i &A)
    {
        SmVecCuda_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }


    __device__
    float l2norm()
    {
        float result = 0;
        for (int i = 0; i < _vecSize; i++)
            result += Vec[i] * Vec[i];
        return sqrt(result);
    }

    __device__
    float l2norm_sqrd()
    {
        float result = 0;
        for (int i = 0; i < _vecSize; i++)
            result += Vec[i] * Vec[i];
        return result;
    }

    __device__
    float l1norm()
    {
        float result = 0;
        for (int i = 0; i < _vecSize; ++i)
            result += AbsVal(Vec[i]);
        return result;
    }

    __device__
    float inf_norm()
    {
        float result = 0;
        for (int i = 0; i < _vecSize; ++i)
            result = result < AbsVal(Vec[i]) ? AbsVal(Vec[i]) : result;
        return result;
    }

    __device__
    float inner_prod(const SmVecCuda_i &y)
    {
        float result = 0;
        for (int i = 0; i < _vecSize; i++)
            result += Vec[i] * y.Vec[i];
        return result;
    }

};

#endif  // MATRIX_H_


