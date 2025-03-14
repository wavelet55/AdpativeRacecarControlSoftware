/************************************************************************
  			BlobResult.h
  			
FUNCIONALITAT: Definici?de la classe CBlobResult
AUTOR: Inspecta S.L.
MODIFICACIONS (Modificaci? Autor, Data):

FUNCTIONALITY: Definition of the CBlobResult class
AUTHOR: Inspecta S.L.
MODIFICATIONS (Modification, Author, Date):

**************************************************************************/

// Disable warning C4819: The file contains a character that cannot be represented in the current code page (949). 
// Save the file in Unicode format to prevent data loss ...
#pragma warning(disable: 4819)

#if !defined(_CUDA_CLASSE_BLOBRESULT_INCLUDED)
#define _CUDA_CLASSE_BLOBRESULT_INCLUDED

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include "BlobLibraryConfiguration.h"
#include <math.h>
//#include "cxcore.h" 
#include <opencv2/core.hpp>

#ifdef MATRIXCV_ACTIU
#include "matrixCV.h"
#else
// llibreria STL
#include "vector"

//! Vector de doubles
typedef std::vector<double> double_stl_vector;
#endif

#include <vector>		// vectors de la STL
#include <functional>
#include "Blob.h"



/**************************************************************************
	Filtres / Filters
**************************************************************************/

//! accions que es poden fer amb els filtres
//! Actions performed by a filter (include or exclude blobs)
#define B_INCLUDE                1L
#define B_EXCLUDE                2L

//! condicions sobre els filtres
//! Conditions to apply the filters
#define B_EQUAL                    3L
#define B_NOT_EQUAL                4L
#define B_GREATER                5L
#define B_LESS                    6L
#define B_GREATER_OR_EQUAL        7L
#define B_LESS_OR_EQUAL            8L
#define B_INSIDE                9L
#define B_OUTSIDE                10L


/**************************************************************************
	Excepcions / Exceptions
**************************************************************************/

//! Excepcions llen?des per les funcions:
#define EXCEPTION_BLOB_OUT_OF_BOUNDS    1000
#define EXCEPCIO_CALCUL_BLOBS            1001


namespace CudaBlobTargetDetectorNS
{

//! definici?de que es un vector de blobs
    typedef std::vector<CBlob_cuda *> blob_vector;

/** 
	Classe que cont?un conjunt de blobs i permet extreure'n propietats 
	o filtrar-los segons determinats criteris.
	Class to calculate the blobs of an image and calculate some properties 
	on them. Also, the class provides functions to filter the blobs using
	some criteria.
*/
    class CBlobResult_cuda
    {
    public:

        //! constructor estandard, crea un conjunt buit de blobs
        //! Standard constructor, it creates an empty set of blobs
        CBlobResult_cuda();

        //! constructor a partir d'una imatge
        //! Image constructor, it creates an object with the blobs of the image
        CBlobResult_cuda(cv::Mat *source, cv::Mat *mask, int threshold, bool findmoments);

        //! constructor de c?ia
        //! Copy constructor
        CBlobResult_cuda(const CBlobResult_cuda &source);

        //! Destructor
        virtual ~CBlobResult_cuda();

        //! operador = per a fer assignacions entre CBlobResult
        //! Assigment operator
        CBlobResult_cuda &operator=(const CBlobResult_cuda &source);

        //! operador + per concatenar dos CBlobResult
        //! Addition operator to concatenate two sets of blobs
        CBlobResult_cuda operator+(const CBlobResult_cuda &source);

        //! Afegeix un blob al conjunt
        //! Adds a blob to the set of blobs
        void AddBlob(CBlob_cuda *blob);

#ifdef MATRIXCV_ACTIU
        //! Calcula un valor sobre tots els blobs de la classe retornant una MatrixCV
        //! Computes some property on all the blobs of the class
        double_vector GetResult( funcio_calculBlob *evaluador ) const;
#endif

        //! Calcula un valor sobre tots els blobs de la classe retornant un std::vector<double>
        //! Computes some property on all the blobs of the class
        double_stl_vector GetSTLResult(funcio_calculBlob *evaluador) const;

        //! Calcula un valor sobre un blob de la classe
        //! Computes some property on one blob of the class
        double GetNumber(int indexblob, funcio_calculBlob *evaluador) const;

        //! Retorna aquells blobs que compleixen les condicions del filtre en el destination
        //! Filters the blobs of the class using some property
        void Filter(CBlobResult_cuda &dst,
                    int filterAction, funcio_calculBlob *evaluador,
                    int condition, double lowLimit, double highLimit = 0);

        //! Retorna l'en?sim blob segons un determinat criteri
        //! Sorts the blobs of the class acording to some criteria and returns the n-th blob
        void GetNthBlob(funcio_calculBlob *criteri, int nBlob, CBlob_cuda &dst) const;

        //! Retorna el blob en?sim
        //! Gets the n-th blob of the class ( without sorting )
        CBlob_cuda GetBlob(int indexblob) const;

        CBlob_cuda *GetBlob(int indexblob);

        //! Elimina tots els blobs de l'objecte
        //! Clears all the blobs of the class
        void ClearBlobs();

        //! Escriu els blobs a un fitxer
        //! Prints some features of all the blobs in a file
        void PrintBlobs(char *nom_fitxer) const;


//Metodes GET/SET

        //! Retorna el total de blobs
        //! Gets the total number of blobs
        int GetNumBlobs() const
        {
            return (m_blobs.size());
        }


    private:

        //! Funci?per gestionar els errors
        //! Function to manage the errors
        void RaiseError(const int errorCode) const;

    protected:

        //! Vector amb els blobs
        //! Vector with all the blobs
        blob_vector m_blobs;
    };

}

#endif // !defined(_CLASSE_BLOBRESULT_INCLUDED)
