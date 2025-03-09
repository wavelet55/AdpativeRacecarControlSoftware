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



#ifndef    SMVECTOR_H_
#define    SMVECTOR_H_

#include <iostream>
#include <string>
#include <math.h>

#pragma warning(disable:4270)

class SmVec_d;

class SmVec_i;

const int MAX_VECTOR_SIZE = 3;


/***************************************/
// Matrix Class for type float

class SmVec_f
{

private:
    int _vecSize;

public:
    float Vec[MAX_VECTOR_SIZE];

    int VecSize()
    {
        return _vecSize;
    }

    inline SmVec_f(int nrows = 1)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE ? MAX_VECTOR_SIZE : nrows;
    }

    inline SmVec_f(int nrows, float ival)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE ? MAX_VECTOR_SIZE : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ival;
    }

    inline SmVec_f(int nrows, const float *ivals)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE ? MAX_VECTOR_SIZE : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ivals[i];
    }

    inline SmVec_f(const SmVec_f &A)        // copy constructor
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];
    }

    ~SmVec_f()
    {}                // destructor


    inline float &val(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    inline float getVal(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    inline SmVec_f operator=(const SmVec_f &A)
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];

        return *this;
    }

    //Make a copy of this...
    //Used for Compatiblity
    inline SmVec_f copy()
    {
        return *this;
    }

    SmVec_i mk_integer();

    SmVec_d mk_double();

    inline SmVec_f operator+(const SmVec_f &A)
    {
        SmVec_f sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] + A.Vec[i];

        return sum;
    }

    inline void operator+=(const SmVec_f &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    inline SmVec_f operator-(const SmVec_f &A)
    {
        SmVec_f sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] - A.Vec[i];

        return sum;
    }

    inline void operator-=(const SmVec_f &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    inline SmVec_f operator-()
    {
        SmVec_f sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = -Vec[i];

        return sum;
    }

    //Note: this is an element by element product... not the
    //inner product of two vectors.
    SmVec_f operator*(const SmVec_f &A)
    {
        SmVec_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }

    //Note: this is an element by element divide.
    SmVec_f operator/(const SmVec_f &A)
    {
        SmVec_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }


    inline SmVec_f operator*(const float x)
    {
        SmVec_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = x * Vec[i];
        return outV;
    }

    inline void operator*=(const float x)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] *= x;
    }

    inline friend SmVec_f operator*(const float x, const SmVec_f &A)
    {
        SmVec_f outV(A._vecSize);
        for (int i = 0; i < A._vecSize; i++)
            outV.Vec[i] = x * A.Vec[i];
        return outV;
    }


    // The next few routines are relational... A < B iff A[i][j] < B[i][j] for all i & j.
    // The routines return a 1 if true a zero otherwise.
    //int	operator<(const SmVec_f &B); 
    //int	operator<=(const SmVec_f &B); 
    //int	operator>(const SmVec_f &B); 
    //int	operator>=(const SmVec_f &B); 
    //int	operator==(const SmVec_f &B); 
    //int	operator!=(const SmVec_f &B); 

    // The next few routines work on every element of the matrix using the 
    // same scaler value:

    inline SmVec_f operator+(const float a)    // Adds "a" to every element of the matrix.
    {
        SmVec_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] + a;
        return outV;
    }

    inline void operator+=(const float a)   // Adds "a" to every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += a;
    }


    inline SmVec_f operator-(const float a)    // Subt "a" from every element of the matrix.
    {
        SmVec_f outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] - a;
        return outV;
    }


    inline void operator-=(const float a)   // Subt "a" from every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] -= a;
    }


    inline void operator=(const float a)    // Sets every element of the matrix to "a".
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = a;
    }


    SmVec_f element_product(const SmVec_f &A);

    //Note: this is an element by element divide.
    SmVec_f element_divide(const SmVec_f &A);

    float l2norm();

    //
    float l2norm_sqrd();

    float l1norm();

    //
    float inf_norm();

    float inner_prod(const SmVec_f &y);

    void print(const std::string &mess);
};

class SmVec_d
{

private:
    int _vecSize;

public:
    double Vec[MAX_VECTOR_SIZE];

    int VecSize()
    {
        return _vecSize;
    }

    inline SmVec_d(int nrows = 1)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE ? MAX_VECTOR_SIZE : nrows;
    }

    inline SmVec_d(int nrows, double ival)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE ? MAX_VECTOR_SIZE : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ival;
    }

    inline SmVec_d(int nrows, const double *ivals)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE ? MAX_VECTOR_SIZE : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ivals[i];
    }

    inline SmVec_d(const SmVec_d &A)        // copy constructor
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];
    }

    ~SmVec_d()
    {}                // destructor


    inline double &val(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    inline double getVal(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    inline SmVec_d operator=(const SmVec_d &A)
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];

        return *this;
    }

    //Make a copy of this...
    //Used for Compatiblity
    inline SmVec_d copy()
    {
        return *this;
    }

    SmVec_i mk_integer();  // create a copy which is of type int...

    SmVec_f mk_float();

    inline SmVec_d operator+(const SmVec_d &A)
    {
        SmVec_d sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] + A.Vec[i];

        return sum;
    }

    inline void operator+=(const SmVec_d &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    inline SmVec_d operator-(const SmVec_d &A)
    {
        SmVec_d sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] - A.Vec[i];

        return sum;
    }

    inline void operator-=(const SmVec_d &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    inline SmVec_d operator-()
    {
        SmVec_d sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = -Vec[i];

        return sum;
    }

    //Note: this is an element by element product... not the
    //inner product of two vectors.
    SmVec_d operator*(const SmVec_d &A)
    {
        SmVec_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }

    //Note: this is an element by element divide.
    SmVec_d operator/(const SmVec_d &A)
    {
        SmVec_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }


    inline SmVec_d operator*(const double x)
    {
        SmVec_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = x * Vec[i];
        return outV;
    }

    inline void operator*=(const double x)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] *= x;
    }

    inline friend SmVec_d operator*(const double x, const SmVec_d &A)
    {
        SmVec_d outV(A._vecSize);
        for (int i = 0; i < A._vecSize; i++)
            outV.Vec[i] = x * A.Vec[i];
        return outV;
    }


    // The next few routines are relational... A < B iff A[i][j] < B[i][j] for all i & j.
    // The routines return a 1 if true a zero otherwise.
    //int	operator<(const SmVec_d &B); 
    //int	operator<=(const SmVec_d &B); 
    //int	operator>(const SmVec_d &B); 
    //int	operator>=(const SmVec_d &B); 
    //int	operator==(const SmVec_d &B); 
    //int	operator!=(const SmVec_d &B); 

    // The next few routines work on every element of the matrix using the 
    // same scaler value:

    inline SmVec_d operator+(const double a)    // Adds "a" to every element of the matrix.
    {
        SmVec_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] + a;
        return outV;
    }

    inline void operator+=(const double a)   // Adds "a" to every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += a;
    }


    inline SmVec_d operator-(const double a)    // Subt "a" from every element of the matrix.
    {
        SmVec_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] - a;
        return outV;
    }


    inline void operator-=(const double a)   // Subt "a" from every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] -= a;
    }


    inline void operator=(const double a)    // Sets every element of the matrix to "a".
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = a;
    }


    inline SmVec_d element_product(const SmVec_d &A)
    {
        SmVec_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }

    //Note: this is an element by element divide.
    inline SmVec_d element_divide(const SmVec_d &A)
    {
        SmVec_d outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }


    double l2norm();

    double l2norm_sqrd();

    double l1norm();

    //
    double inf_norm();

    double inner_prod(const SmVec_d &y);

    void print(const std::string &mess);
};

class SmVec_i
{

private:
    int _vecSize;

public:
    int Vec[MAX_VECTOR_SIZE];

    int VecSize()
    {
        return _vecSize;
    }

    inline SmVec_i(int nrows = 1)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE ? MAX_VECTOR_SIZE : nrows;
    }

    inline SmVec_i(int nrows, int ival)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE ? MAX_VECTOR_SIZE : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ival;
    }

    inline SmVec_i(int nrows, const int *ivals)
    {
        _vecSize = nrows < 1 ? 1 : nrows > MAX_VECTOR_SIZE ? MAX_VECTOR_SIZE : nrows;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = ivals[i];
    }

    inline SmVec_i(const SmVec_i &A)        // copy constructor
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];
    }

    ~SmVec_i()
    {}                // destructor


    inline int &val(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    inline int getVal(const int row)
    {
        int r = row < 0 ? 0 : row >= _vecSize ? _vecSize - 1 : row;
        return (Vec[r]);
    }

    inline SmVec_i operator=(const SmVec_i &A)
    {
        _vecSize = A._vecSize;
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = A.Vec[i];

        return *this;
    }

    //Make a copy of this...
    //Used for Compatiblity
    inline SmVec_i copy()
    {
        return *this;
    }

    SmVec_d mk_double();  // create a copy which is of type int...

    SmVec_f mk_float();  // create a copy which is of type integer...

    inline SmVec_i operator+(const SmVec_i &A)
    {
        SmVec_i sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] + A.Vec[i];

        return sum;
    }

    inline void operator+=(const SmVec_i &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    inline SmVec_i operator-(const SmVec_i &A)
    {
        SmVec_i sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = Vec[i] - A.Vec[i];

        return sum;
    }

    inline void operator-=(const SmVec_i &A)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += A.Vec[i];
    }

    inline SmVec_i operator-()
    {
        SmVec_i sum(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            sum.Vec[i] = -Vec[i];

        return sum;
    }

    //Note: this is an element by element product... not the
    //inner product of two vectors.
    SmVec_i operator*(const SmVec_i &A)
    {
        SmVec_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }

    //Note: this is an element by element divide.
    SmVec_i operator/(const SmVec_i &A)
    {
        SmVec_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }


    inline SmVec_i operator*(const int x)
    {
        SmVec_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = x * Vec[i];
        return outV;
    }

    inline void operator*=(const int x)
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] *= x;
    }

    inline friend SmVec_i operator*(const int x, const SmVec_i &A)
    {
        SmVec_i outV(A._vecSize);
        for (int i = 0; i < A._vecSize; i++)
            outV.Vec[i] = x * A.Vec[i];
        return outV;
    }


    // The next few routines are relational... A < B iff A[i][j] < B[i][j] for all i & j.
    // The routines return a 1 if true a zero otherwise.
    //int	operator<(const SmVec_i &B); 
    //int	operator<=(const SmVec_i &B); 
    //int	operator>(const SmVec_i &B); 
    //int	operator>=(const SmVec_i &B); 
    //int	operator==(const SmVec_i &B); 
    //int	operator!=(const SmVec_i &B); 

    // The next few routines work on every element of the matrix using the 
    // same scaler value:

    inline SmVec_i operator+(const int a)    // Adds "a" to every element of the matrix.
    {
        SmVec_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] + a;
        return outV;
    }

    inline void operator+=(const int a)   // Adds "a" to every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] += a;
    }


    inline SmVec_i operator-(const int a)    // Subt "a" from every element of the matrix.
    {
        SmVec_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] - a;
        return outV;
    }


    inline void operator-=(const int a)   // Subt "a" from every element of the matrix.
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] -= a;
    }


    inline void operator=(const int a)    // Sets every element of the matrix to "a".
    {
        for (int i = 0; i < _vecSize; i++)
            Vec[i] = a;
    }


    inline SmVec_i element_product(const SmVec_i &A)
    {
        SmVec_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] * A.Vec[i];
        return outV;
    }

    //Note: this is an element by element divide.
    inline SmVec_i element_divide(const SmVec_i &A)
    {
        SmVec_i outV(_vecSize);
        for (int i = 0; i < _vecSize; i++)
            outV.Vec[i] = Vec[i] / A.Vec[i];
        return outV;
    }


    double l2norm();

    double l2norm_sqrd();

    double l1norm();

    double inf_norm();

    double inner_prod(const SmVec_i &y);

    void print(const std::string &mess);
};

#endif  // MATRIX_H_


