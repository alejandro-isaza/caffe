#pragma once
#include <Accelerate/Accelerate.h>


namespace caffe {
namespace dsp {

/**
 Used for numbers of elements in arrays and for indices of elements in arrays.
 (It is also used for the base-two logarithm of numbers of elements, although
 a much smaller type is suitable for that.)
 */
using Length = ::vDSP_Length;

/**
 Used for differences of indices of elements (which of course includes
 strides).
 */
using Stride = ::vDSP_Stride;

/**
 Unsigned 24-bit integer.
 */
using uint24 = ::vDSP_uint24;

/**
 Signed 24-bit integer.
 */
using int24 = ::vDSP_int24;


template <typename T>
struct ComplexType {
    using type = void;
};
template <>
struct ComplexType<float> {
    using type = ::DSPComplex;
};
template <>
struct ComplexType<double> {
    using type = ::DSPDoubleComplex;
};

/**
 Used to hold a complex value
 */
template <typename T>
using Complex = typename ComplexType<T>::type;


template <typename T>
struct SplitComplexType {
    using type = void;
};
template <>
struct SplitComplexType<float> {
    using type = ::DSPSplitComplex;
};
template <>
struct SplitComplexType<double> {
    using type = ::DSPDoubleSplitComplex;
};

/**
 Used to represent a complex number when the real and imaginary parts are
 stored in separate arrays.
 */
template <typename T>
using SplitComplex = typename SplitComplexType<T>::type;


#pragma mark - FFT

enum class FFTDirection {
    /// Specify a forward transform
    Forward = kFFTDirection_Forward,

    /// Specify an inverse transform
    Inverse = kFFTDirection_Inverse
};

enum class FFTRadix {
    Radix2 = kFFTRadix2,
    Radix3 = kFFTRadix3,
    Radix5 = kFFTRadix5
};

//enum {
//    vDSP_HALF_WINDOW              = 1,
//    vDSP_HANN_DENORM              = 0,
//    vDSP_HANN_NORM                = 2
//};



template <typename T>
struct FFTSetupType {
    using type = void;
};
template <>
struct FFTSetupType<float> {
    using type = ::FFTSetup;
};
template <>
struct FFTSetupType<double> {
    using type = ::FFTSetupD;
};

/// An opaque type that contains setup information for a given FFT transform.
template <typename T>
using FFTSetup = typename FFTSetupType<T>::type;


/**
 create_fftsetup allocates memory and prepares constants used by FFT routines.
 */
template <typename T>
FFTSetup<T> create_fftsetup(Length log2n, FFTRadix radix) noexcept;

template <>
inline FFTSetup<float> create_fftsetup<float>(Length log2n, FFTRadix radix) noexcept {
    return ::vDSP_create_fftsetup(log2n, static_cast<::FFTRadix>(radix));
}

template <>
inline FFTSetup<double> create_fftsetup<double>(Length log2n, FFTRadix radix) noexcept {
    return ::vDSP_create_fftsetupD(log2n, static_cast<::FFTRadix>(radix));
}

/**
 destroy_fftsetup frees the memory. It may be passed a null pointer, in which
 case it has no effect.
 */
inline void destroy_fftsetup(FFTSetup<float> setup) noexcept {
    ::vDSP_destroy_fftsetup(setup);
}
inline void destroy_fftsetup(FFTSetup<double> setup) noexcept {
    ::vDSP_destroy_fftsetupD(setup);
}


/// Convert a complex array to a complex-split array.
inline void ctoz(const Complex<float> *C,
                 Stride IC,
                 const SplitComplex<float> *Z,
                 Stride IZ,
                 Length N) noexcept {
    ::vDSP_ctoz(C, IC, Z, IZ, N);
}

/// Convert a complex array to a complex-split array.
inline void ctoz(const Complex<double> *C,
                 Stride IC,
                 const SplitComplex<double> *Z,
                 Stride IZ,
                 Length N) noexcept {
    ::vDSP_ctozD(C, IC, Z, IZ, N);
}

/*  Map:

 Pseudocode:     Memory:
 C[n]            C[n*IC/2].real + i * C[n*IC/2].imag
 Z[n]            Z->realp[n*IZ] + i * Z->imagp[n*IZ]

 These compute:

 for (n = 0; n < N; ++n)
 Z[n] = C[n];
 */


/// Convert a complex-split array to a complex array.
inline void ztoc(const SplitComplex<float> *Z,
                 Stride IZ,
                 Complex<float> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_ztoc(Z, IZ, C, IC, N);
}
inline void ztoc(const SplitComplex<double> *Z,
                 Stride IZ,
                 Complex<double> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_ztocD(Z, IZ, C, IC, N);
}

/*  Map:

 Pseudocode:     Memory:
 Z[n]            Z->realp[n*IZ] + i * Z->imagp[n*IZ]
 C[n]            C[n*IC/2].real + i * C[n*IC/2].imag

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = Z[n];
 */



/**
 In-place complex Discrete Fourier Transform routines, with and without
 temporary memory.  We suggest you use the DFT routines instead of these.
 */
inline void fft_zip(FFTSetup<float> Setup, const SplitComplex<float> *C, Stride IC, Length Log2N, FFTDirection Direction) noexcept {
    ::vDSP_fft_zip(Setup, C, IC, Log2N, static_cast<::FFTDirection>(Direction));
}
inline void fft_zip(FFTSetup<double> Setup, const SplitComplex<double> *C, Stride IC, Length Log2N, FFTDirection Direction) noexcept {
    ::vDSP_fft_zipD(Setup, C, IC, Log2N, static_cast<::FFTDirection>(Direction));
}
inline void fft_zipt(FFTSetup<float> Setup,
                     const SplitComplex<float> *C,
                     Stride IC,
                     const SplitComplex<float> *Buffer,
                     Length Log2N,
                     FFTDirection Direction) noexcept {
    return ::vDSP_fft_zipt(Setup, C, IC, Buffer, Log2N, static_cast<::FFTDirection>(Direction));
}

inline void fft_zipt(FFTSetup<double> Setup,
                     const SplitComplex<double> *C,
                     Stride IC,
                     const SplitComplex<double> *Buffer,
                     Length Log2N,
                     FFTDirection Direction) noexcept {
    return ::vDSP_fft_ziptD(Setup, C, IC, Buffer, Log2N, static_cast<::FFTDirection>(Direction));
}
/*  Maps:

 For this routine, strides are shown explicitly; the default maps
 are not used.

 These compute:

 N = 1 << Log2N;

 scale = 0 < Direction ? 1 : 1./N;

 // Define a complex vector, h:
 for (j = 0; j < N; ++j)
 h[j] = C->realp[j*IC] + i * C->imagp[j*IC];

 // Perform Discrete Fourier Transform.
 for (k = 0; k < N; ++k)
 H[k] = scale * sum(h[j] * e**(-Direction*2*pi*i*j*k/N), 0 <= j < N);

 // Store result.
 for (k = 0; k < N; ++k)
 {
 C->realp[k*IC] = Re(H[k]);
 C->imagp[k*IC] = Im(H[k]);
 }

 Setup must have been properly created by a call to create_fftsetup
 and not subsequently destroyed.

 Direction must be +1 or -1.

 The temporary buffer versions perform the same operation but are
 permitted to use the temporary buffer for improved performance.  Each
 of Buffer->realp and Buffer->imagp must contain the lesser of 16,384
 bytes or N * sizeof *C->realp bytes and is preferably 16-byte aligned
 or better.
 */


/**
 Out-of-place complex Discrete Fourier Transform routines, with and without
 temporary memory.  We suggest you use the DFT routines instead of these.
 */
inline void fft_zop(FFTSetup<float> Setup,
                    const SplitComplex<float> *A,
                    Stride IA,
                    const SplitComplex<float> *C,
                    Stride IC,
                    Length Log2N,
                    FFTDirection Direction) noexcept {
    ::vDSP_fft_zop(Setup, A, IA, C, IC, Log2N, static_cast<::FFTDirection>(Direction));
}

inline void fft_zopt(FFTSetup<float> Setup,
                     const SplitComplex<float> *A,
                     Stride IA,
                     const SplitComplex<float> *C,
                     Stride IC,
                     const SplitComplex<float> *Buffer,
                     Length Log2N,
                     FFTDirection Direction) noexcept {
    ::vDSP_fft_zopt(Setup, A, IA, C, IC, Buffer, Log2N, static_cast<::FFTDirection>(Direction));
}
inline void fft_zop(FFTSetup<double> Setup,
                    const SplitComplex<double> *A,
                    Stride IA,
                    const SplitComplex<double> *C,
                    Stride IC,
                    Length Log2N,
                    FFTDirection Direction) noexcept {
    ::vDSP_fft_zopD(Setup, A, IA, C, IC, Log2N, static_cast<::FFTDirection>(Direction));
}
inline void fft_zopt(FFTSetup<double> Setup,
                     const SplitComplex<double> *A,
                     Stride IA,
                     const SplitComplex<double> *C,
                     Stride IC,
                     const SplitComplex<double> *Buffer,
                     Length Log2N,
                     FFTDirection Direction) noexcept {
    ::vDSP_fft_zoptD(Setup, A, IA, C, IC, Buffer, Log2N, static_cast<::FFTDirection>(Direction));
}
/*  Maps:

 For this routine, strides are shown explicitly; the default maps
 are not used.

 These compute:

 N = 1 << Log2N;

 scale = 0 < Direction ? 1 : 1./N;

 // Define a complex vector, h:
 for (j = 0; j < N; ++j)
 h[j] = A->realp[j*IA] + i * A->imagp[j*IA];

 // Perform Discrete Fourier Transform.
 for (k = 0; k < N; ++k)
 H[k] = scale * sum(h[j] * e**(-Direction*2*pi*i*j*k/N), 0 <= j < N);

 // Store result.
 for (k = 0; k < N; ++k)
 {
 C->realp[k*IC] = Re(H[k]);
 C->imagp[k*IC] = Im(H[k]);
 }

 Setup must have been properly created by a call to create_fftsetup
 and not subsequently destroyed.

 Direction must be +1 or -1.

 The temporary buffer versions perform the same operation but are
 permitted to use the temporary buffer for improved performance.  Each
 of Buffer->realp and Buffer->imagp must contain the lesser of 16,384
 bytes or N * sizeof *C->realp bytes and is preferably 16-byte aligned
 or better.
 */


/**
 In-place real-to-complex Discrete Fourier Transform routines, with and
 without temporary memory.  We suggest you use the DFT routines instead of
 these.
 */
inline void fft_zrip(
                     FFTSetup<float> Setup,
                     const SplitComplex<float> *C,
                     Stride IC,
                     Length Log2N,
                     FFTDirection Direction) noexcept {
    ::vDSP_fft_zrip(Setup, C, IC, Log2N, static_cast<::FFTDirection>(Direction));
}
inline void fft_zrip(
                     FFTSetup<double> Setup,
                     const SplitComplex<double> *C,
                     Stride IC,
                     Length Log2N,
                     FFTDirection Direction) noexcept {
    ::vDSP_fft_zripD(Setup, C, IC, Log2N, static_cast<::FFTDirection>(Direction));
}
inline void fft_zript(
                      FFTSetup<float> Setup,
                      const SplitComplex<float> *C,
                      Stride IC,
                      const SplitComplex<float> *Buffer,
                      Length Log2N,
                      FFTDirection Direction) noexcept {
    ::vDSP_fft_zript(Setup, C, IC, Buffer, Log2N, static_cast<::FFTDirection>(Direction));
}
inline void fft_zript(
                      FFTSetup<double> Setup,
                      const SplitComplex<double> *C,
                      Stride IC,
                      const SplitComplex<double> *Buffer,
                      Length Log2N,
                      FFTDirection Direction) noexcept {
    ::vDSP_fft_zriptD(Setup, C, IC, Buffer, Log2N, static_cast<::FFTDirection>(Direction));
}
/*  Maps:

 For this routine, strides are shown explicitly; the default maps
 are not used.

 These compute:

 N = 1 << Log2N;

 If Direction is +1, a real-to-complex transform is performed, taking
 input from a real vector that has been coerced into the complex
 structure:

 scale = 2;

 // Define a real vector, h:
 for (j = 0; j < N/2; ++j)
 {
 h[2*j + 0] = C->realp[j*IC];
 h[2*j + 1] = C->imagp[j*IC];
 }

 // Perform Discrete Fourier Transform.
 for (k = 0; k < N; ++k)
 H[k] = scale *
 sum(h[j] * e**(-Direction*2*pi*i*j*k/N), 0 <= j < N);

 // Pack DC and Nyquist components into C->realp[0] and C->imagp[0].
 C->realp[0*IC] = Re(H[ 0 ]).
 C->imagp[0*IC] = Re(H[N/2]).

 // Store regular components:
 for (k = 1; k < N/2; ++k)
 {
 C->realp[k*IC] = Re(H[k]);
 C->imagp[k*IC] = Im(H[k]);
 }

 Note that, for N/2 < k < N, H[k] is not stored.  However, since
 the input is a real vector, the output has symmetry that allows the
 unstored elements to be derived from the stored elements:  H[k] =
 conj(H(N-k)).  This symmetry also implies the DC and Nyquist
 components are real, so their imaginary parts are zero.

 If Direction is -1, a complex-to-real inverse transform is performed,
 producing a real output vector coerced into the complex structure:

 scale = 1./N;

 // Define a complex vector, h:
 h[ 0 ] = C->realp[0*IC];
 h[N/2] = C->imagp[0*IC];
 for (j = 1; j < N/2; ++j)
 {
 h[ j ] = C->realp[j*IC] + i * C->imagp[j*IC];
 h[N-j] = conj(h[j]);
 }

 // Perform Discrete Fourier Transform.
 for (k = 0; k < N; ++k)
 H[k] = scale *
 sum(h[j] * e**(-Direction*2*pi*i*j*k/N), 0 <= j < N);

 // Coerce real results into complex structure:
 for (k = 0; k < N/2; ++k)
 {
 C->realp[k*IC] = H[2*k+0];
 C->imagp[k*IC] = H[2*k+1];
 }

 Note that, mathematically, the symmetry in the input vector compels
 every H[k] to be real, so there are no imaginary components to be
 stored.

 Setup must have been properly created by a call to create_fftsetup
 and not subsequently destroyed.

 Direction must be +1 or -1.

 The temporary buffer versions perform the same operation but are
 permitted to use the temporary buffer for improved performance.  Each
 of Buffer->realp and Buffer->imagp must contain N/2 * sizeof *C->realp
 bytes and is preferably 16-byte aligned or better.
 */


/**
 Out-of-place real-to-complex Discrete Fourier Transform routines, with and
 without temporary memory.  We suggest you use the DFT routines instead of
 these.
 */
inline void fft_zrop(
                     FFTSetup<float> Setup,
                     const SplitComplex<float> *A,
                     Stride IA,
                     const SplitComplex<float> *C,
                     Stride IC,
                     Length Log2N,
                     FFTDirection Direction) noexcept {
    ::vDSP_fft_zrop(Setup, A, IA, C, IC, Log2N, static_cast<::FFTDirection>(Direction));
}
inline void fft_zrop(
                     FFTSetup<double> Setup,
                     const SplitComplex<double> *A,
                     Stride IA,
                     const SplitComplex<double> *C,
                     Stride IC,
                     Length Log2N,
                     FFTDirection Direction) noexcept {
    ::vDSP_fft_zropD(Setup, A, IA, C, IC, Log2N, static_cast<::FFTDirection>(Direction));
}
inline void fft_zropt(
                      FFTSetup<float> Setup,
                      const SplitComplex<float> *A,
                      Stride IA,
                      const SplitComplex<float> *C,
                      Stride IC,
                      const SplitComplex<float> *Buffer,
                      Length Log2N,
                      FFTDirection Direction) noexcept {
    ::vDSP_fft_zropt(Setup, A, IA, C, IC, Buffer, Log2N, static_cast<::FFTDirection>(Direction));
}
inline void fft_zropt(
                      FFTSetup<double> Setup,
                      const SplitComplex<double> *A,
                      Stride IA,
                      const SplitComplex<double> *C,
                      Stride IC,
                      const SplitComplex<double> *Buffer,
                      Length Log2N,
                      FFTDirection Direction) noexcept {
    ::vDSP_fft_zroptD(Setup, A, IA, C, IC, Buffer, Log2N, static_cast<::FFTDirection>(Direction));
}


/*  Matrix multiply.
 */
inline void mmul(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 float *C,
                 Stride IC,
                 Length M,
                 Length N,
                 Length P) noexcept {
    ::vDSP_mmul(A, IA, B, IB, C, IC, M, N, P);
}
inline void mmul(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 double *C,
                 Stride IC,
                 Length M,
                 Length N,
                 Length P) noexcept {
    ::vDSP_mmulD(A, IA, B, IB, C, IC, M, N, P);
}
/*  Maps:

 A is regarded as a two-dimensional matrix with dimemnsions [M][P]
 and stride IA.  B is regarded as a two-dimensional matrix with
 dimemnsions [P][N] and stride IB.  C is regarded as a
 two-dimensional matrix with dimemnsions [M][N] and stride IC.

 Pseudocode:     Memory:
 A[m][p]         A[(m*P+p)*IA]
 B[p][n]         B[(p*N+n)*IB]
 C[m][n]         C[(m*N+n)*IC]

 These compute:

 for (m = 0; m < M; ++m)
 for (n = 0; n < N; ++n)
 C[m][n] = sum(A[m][p] * B[p][n], 0 <= p < P);
 */


/*  Split-complex matrix multiply and add.
 */
inline void zmma(
                 const SplitComplex<float> *A,
                 Stride IA,
                 const SplitComplex<float> *B,
                 Stride IB,
                 const SplitComplex<float> *C,
                 Stride IC,
                 const SplitComplex<float> *D,
                 Stride ID,
                 Length M,
                 Length N,
                 Length P) noexcept {
    ::vDSP_zmma(A, IA, B, IB, C, IC, D, ID, M, N, P);
}
inline void zmma(
                 const SplitComplex<double> *A,
                 Stride IA,
                 const SplitComplex<double> *B,
                 Stride IB,
                 const SplitComplex<double> *C,
                 Stride IC,
                 const SplitComplex<double> *D,
                 Stride ID,
                 Length M,
                 Length N,
                 Length P) noexcept {
    ::vDSP_zmmaD(A, IA, B, IB, C, IC, D, ID, M, N, P);
}
/*  Maps:

 Pseudocode:     Memory:
 A[m][p]         A->realp[(m*P+p)*IA] + i * A->imagp[(m*P+p)*IA].
 B[p][n]         B->realp[(p*N+n)*IB] + i * B->imagp[(p*N+n)*IB].
 C[m][n]         C->realp[(m*N+n)*IC] + i * C->imagp[(m*N+n)*IC].
 D[m][n]         D->realp[(m*N+n)*ID] + i * D->imagp[(m*N+n)*ID].

 These compute:

 for (m = 0; m < M; ++m)
 for (n = 0; n < N; ++n)
 D[m][n] = sum(A[m][p] * B[p][n], 0 <= p < P) + C[m][n];
 */


/*  Split-complex matrix multiply and subtract.
 */
inline void zmms(
                 const SplitComplex<float> *A,
                 Stride IA,
                 const SplitComplex<float> *B,
                 Stride IB,
                 const SplitComplex<float> *C,
                 Stride IC,
                 const SplitComplex<float> *D,
                 Stride ID,
                 Length M,
                 Length N,
                 Length P) noexcept {
    ::vDSP_zmms(A, IA, B, IB, C, IC, D, ID, M, N, P);
}
inline void zmms(
                 const SplitComplex<double> *A,
                 Stride IA,
                 const SplitComplex<double> *B,
                 Stride IB,
                 const SplitComplex<double> *C,
                 Stride IC,
                 const SplitComplex<double> *D,
                 Stride ID,
                 Length M,
                 Length N,
                 Length P) noexcept {
    ::vDSP_zmmsD(A, IA, B, IB, C, IC, D, ID, M, N, P);
}
/*  Maps:

 Pseudocode:     Memory:
 A[m][p]         A->realp[(m*P+p)*IA] + i * A->imagp[(m*P+p)*IA].
 B[p][n]         B->realp[(p*N+n)*IB] + i * B->imagp[(p*N+n)*IB].
 C[m][n]         C->realp[(m*N+n)*IC] + i * C->imagp[(m*N+n)*IC].
 D[m][n]         D->realp[(m*N+n)*ID] + i * D->imagp[(m*N+n)*ID].

 These compute:

 for (m = 0; m < M; ++m)
 for (n = 0; n < N; ++n)
 D[m][n] = sum(A[m][p] * B[p][n], 0 <= p < P) - C[m][n];
 */


// Vector multiply, multiply, add, and add.
inline void zvmmaa(
                   const SplitComplex<float> *A,
                   Stride IA,
                   const SplitComplex<float> *B,
                   Stride IB,
                   const SplitComplex<float> *C,
                   Stride IC,
                   const SplitComplex<float> *D,
                   Stride ID,
                   const SplitComplex<float> *E,
                   Stride IE,
                   const SplitComplex<float> *F,
                   Stride IF,
                   Length N) noexcept {
    ::vDSP_zvmmaa(A, IA, B, IB, C, IC, D, ID, E, IE, F, IF, N);

}
inline void zvmmaa(
                   const SplitComplex<double> *A,
                   Stride IA,
                   const SplitComplex<double> *B,
                   Stride IB,
                   const SplitComplex<double> *C,
                   Stride IC,
                   const SplitComplex<double> *D,
                   Stride ID,
                   const SplitComplex<double> *E,
                   Stride IE,
                   const SplitComplex<double> *F,
                   Stride IF,
                   Length N) noexcept {
    ::vDSP_zvmmaaD(A, IA, B, IB, C, IC, D, ID, E, IE, F, IF, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 F[n] = A[n] * B[n] + C[n] * D[n] + E[n];
 */


/*  Split-complex matrix multiply and reverse subtract.
 */
inline void zmsm(
                 const SplitComplex<float> *A,
                 Stride IA,
                 const SplitComplex<float> *B,
                 Stride IB,
                 const SplitComplex<float> *C,
                 Stride IC,
                 const SplitComplex<float> *D,
                 Stride ID,
                 Length M,
                 Length N,
                 Length P) noexcept {
    ::vDSP_zmsm(A, IA, B, IB, C, IC, D, ID, M, N, P);

}
inline void zmsm(
                 const SplitComplex<double> *A,
                 Stride IA,
                 const SplitComplex<double> *B,
                 Stride IB,
                 const SplitComplex<double> *C,
                 Stride IC,
                 const SplitComplex<double> *D,
                 Stride ID,
                 Length M,
                 Length N,
                 Length P) noexcept {
    ::vDSP_zmsmD(A, IA, B, IB, C, IC, D, ID, M, N, P);

}
/*  Maps:

 Pseudocode:     Memory:
 A[m][p]         A->realp[(m*P+p)*IA] + i * A->imagp[(m*P+p)*IA].
 B[p][n]         B->realp[(p*N+n)*IB] + i * B->imagp[(p*N+n)*IB].
 C[m][n]         C->realp[(m*N+n)*IC] + i * C->imagp[(m*N+n)*IC].
 D[m][n]         D->realp[(m*N+n)*ID] + i * D->imagp[(m*N+n)*ID].

 These compute:

 for (m = 0; m < M; ++m)
 for (n = 0; n < N; ++n)
 D[m][n] = C[m][n] - sum(A[m][p] * B[p][n], 0 <= p < P);
 */


/*  Split-complex matrix multiply.
 */
inline void zmmul(
                  const SplitComplex<float> *A,
                  Stride IA,
                  const SplitComplex<float> *B,
                  Stride IB,
                  const SplitComplex<float> *C,
                  Stride IC,
                  Length M,
                  Length N,
                  Length P) noexcept {
    ::vDSP_zmmul(A, IA, B, IB, C, IC, M, N, P);

}
inline void zmmul(
                  const SplitComplex<double> *A,
                  Stride IA,
                  const SplitComplex<double> *B,
                  Stride IB,
                  const SplitComplex<double> *C,
                  Stride IC,
                  Length M,
                  Length N,
                  Length P) noexcept {
    ::vDSP_zmmulD(A, IA, B, IB, C, IC, M, N, P);

}


#pragma mark - Vector operations

/*  Maps:

 Pseudocode:     Memory:
 A[m][p]         A->realp[(m*P+p)*IA] + i * A->imagp[(m*P+p)*IA].
 B[p][n]         B->realp[(p*N+n)*IB] + i * B->imagp[(p*N+n)*IB].
 C[m][n]         C->realp[(m*N+n)*IC] + i * C->imagp[(m*N+n)*IC].

 These compute:

 for (m = 0; m < M; ++m)
 for (n = 0; n < N; ++n)
 C[m][n] = sum(A[m][p] * B[p][n], 0 <= p < P);
 */


// Vector add.
inline void vadd(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vadd(A, IA, B, IB, C, IC, N);

}
inline void vadd(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vaddD(A, IA, B, IB, C, IC, N);

}
inline void vadd(
                 const int *A,
                 Stride IA,
                 const int *B,
                 Stride IB,
                 int *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vaddi(A, IA, B, IB, C, IC, N);

}
inline void vadd(
                 const SplitComplex<float> *A,
                 Stride IA,
                 const SplitComplex<float> *B,
                 Stride IB,
                 const SplitComplex<float> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zvadd(A, IA, B, IB, C, IC, N);

}
inline void vadd(
                 const SplitComplex<double> *A,
                 Stride IA,
                 const SplitComplex<double> *B,
                 Stride IB,
                 const SplitComplex<double> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zvaddD(A, IA, B, IB, C, IC, N);

}
inline void vadd(
                 const SplitComplex<float> *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 const SplitComplex<float> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zrvadd(A, IA, B, IB, C, IC, N);

}
inline void vadd(
                 const SplitComplex<double> *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 const SplitComplex<double> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zrvaddD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] + B[n];
 */


// Vector subtract.
inline void vsub(
                 const float *B,  // Caution:  A and B are swapped!
                 Stride IB,
                 const float *A,  // Caution:  A and B are swapped!
                 Stride IA,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vsub(A, IA, B, IB, C, IC, N);

}
inline void vsub(
                 const double *B, // Caution:  A and B are swapped!
                 Stride IB,
                 const double *A, // Caution:  A and B are swapped!
                 Stride IA,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vsubD(A, IA, B, IB, C, IC, N);

}
inline void vsub(
                 const SplitComplex<float> *A,
                 Stride IA,
                 const SplitComplex<float> *B,
                 Stride IB,
                 const SplitComplex<float> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zvsub(A, IA, B, IB, C, IC, N);

}
inline void vsub(
                 const SplitComplex<double> *A,
                 Stride IA,
                 const SplitComplex<double> *B,
                 Stride IB,
                 const SplitComplex<double> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zvsubD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] - B[n];
 */


// Vector multiply.
inline void vmul(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vmul(A, IA, B, IB, C, IC, N);

}
inline void vmul(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vmulD(A, IA, B, IB, C, IC, N);

}
inline void vmul(
                 const SplitComplex<float> *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 const SplitComplex<float> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zrvmul(A, IA, B, IB, C, IC, N);

}
inline void vmul(
                 const SplitComplex<double> *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 const SplitComplex<double> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zrvmulD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] * B[n];
 */


// Vector divide.
inline void vdiv(
                 const float *B,  // Caution:  A and B are swapped!
                 Stride IB,
                 const float *A,  // Caution:  A and B are swapped!
                 Stride IA,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vdiv(B, IB, A, IA, C, IC, N);

}
inline void vdiv(
                 const double *B, // Caution:  A and B are swapped!
                 Stride IB,
                 const double *A, // Caution:  A and B are swapped!
                 Stride IA,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vdivD(B, IB, A, IA, C, IC, N);

}
inline void vdiv(
                 const int *B,  // Caution:  A and B are swapped!
                 Stride IB,
                 const int *A,  // Caution:  A and B are swapped!
                 Stride IA,
                 int *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vdivi(B, IB, A, IA, C, IC, N);

}
inline void vdiv(
                 const SplitComplex<float> *B,    // Caution:  A and B are swapped!
                 Stride IB,
                 const SplitComplex<float> *A,    // Caution:  A and B are swapped!
                 Stride IA,
                 const SplitComplex<float> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zvdiv(B, IB, A, IA, C, IC, N);

}
inline void vdiv(
                 const SplitComplex<double> *B,  // Caution:  A and B are swapped!
                 Stride IB,
                 const SplitComplex<double> *A,  // Caution:  A and B are swapped!
                 Stride IA,
                 const SplitComplex<double> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zvdivD(B, IB, A, IA, C, IC, N);

}
inline void vdiv(
                 const SplitComplex<float> *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 const SplitComplex<float> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zrvdiv(A, IA, B, IB, C, IC, N);

}
inline void vdiv(
                 const SplitComplex<double> *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 const SplitComplex<double> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zrvdivD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] / B[n];
 */


// Vector-scalar multiply.
inline void vsmul(
                  const float *A,
                  Stride IA,
                  const float *B,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vsmul(A, IA, B, C, IC, N);

}
inline void vsmul(
                  const double *A,
                  Stride IA,
                  const double *B,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vsmulD(A, IA, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] * B[0];
 */


// Vector square.
inline void vsq(
                const float *A,
                Stride IA,
                float *C,
                Stride IC,
                Length N) noexcept {
    ::vDSP_vsq(A, IA, C, IC, N);

}
inline void vsq(
                const double *A,
                Stride IA,
                double *C,
                Stride IC,
                Length N) noexcept {
    ::vDSP_vsqD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n]**2;
 */



// Vector signed square.
inline void vssq(
                 const float *A,
                 Stride IA,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vssq(A, IA, C, IC, N);

}
inline void vssq(
                 const double *A,
                 Stride IA,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vssqD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] * |A[n]|;
 */


// Euclidean distance, squared.
inline void distancesq(
                       const float *A,
                       Stride IA,
                       const float *B,
                       Stride IB,
                       float *C,
                       Length N) noexcept {
    ::vDSP_distancesq(A, IA, B, IB, C, N);

}
inline void distancesq(
                       const double *A,
                       Stride IA,
                       const double *B,
                       Stride IB,
                       double *C,
                       Length N) noexcept {
    ::vDSP_distancesqD(A, IA, B, IB, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum((A[n] - B[n]) ** 2, 0 <= n < N);
 */


// Dot product.
inline void dotpr(
                  const float *A,
                  Stride IA,
                  const float *B,
                  Stride IB,
                  float *C,
                  Length N) noexcept {
    ::vDSP_dotpr(A, IA, B, IB, C, N);

}
inline void dotpr(
                  const double *A,
                  Stride IA,
                  const double *B,
                  Stride IB,
                  double *C,
                  Length N) noexcept {
    ::vDSP_dotprD(A, IA, B, IB, C, N);

}
inline void dotpr(
                  const SplitComplex<float> *A,
                  Stride IA,
                  const SplitComplex<float> *B,
                  Stride IB,
                  const SplitComplex<float> *C,
                  Length N) noexcept {
    ::vDSP_zdotpr(A, IA, B, IB, C, N);

}
inline void dotpr(
                  const SplitComplex<double> *A,
                  Stride IA,
                  const SplitComplex<double> *B,
                  Stride IB,
                  const SplitComplex<double> *C,
                  Length N) noexcept {
    ::vDSP_zdotprD(A, IA, B, IB, C, N);

}
inline void dotpr(
                  const SplitComplex<float> *A,
                  Stride IA,
                  const float *B,
                  Stride IB,
                  const SplitComplex<float> *C,
                  Length N) noexcept {
    ::vDSP_zrdotpr(A, IA, B, IB, C, N);

}
inline void dotpr(
                  const SplitComplex<double> *A,
                  Stride IA,
                  const double *B,
                  Stride IB,
                  const SplitComplex<double> *C,
                  Length N) noexcept {
    ::vDSP_zrdotprD(A, IA, B, IB, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum(A[n] * B[n], 0 <= n < N);
 */


// Vector add and multiply.
inline void vam(
                const float *A,
                Stride IA,
                const float *B,
                Stride IB,
                const float *C,
                Stride IC,
                float *D,
                Stride ID,
                Length N) noexcept {
    ::vDSP_vam(A, IA, B, IB, C, IC, D, ID, N);

}
inline void vam(
                const double *A,
                Stride IA,
                const double *B,
                Stride IB,
                const double *C,
                Stride IC,
                double *D,
                Stride ID,
                Length N) noexcept {
    ::vDSP_vamD(A, IA, B, IB, C, IC, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = (A[n] + B[n]) * C[n];
 */


// Vector multiply and add.
inline void vma(
                const float *A,
                Stride IA,
                const float *B,
                Stride IB,
                const float *C,
                Stride IC,
                float *D,
                Stride ID,
                Length N) noexcept {
    ::vDSP_vma(A, IA, B, IB, C, IC, D, ID, N);

}
inline void vma(
                const double *A,
                Stride IA,
                const double *B,
                Stride IB,
                const double *C,
                Stride IC,
                double *D,
                Stride ID,
                Length N) noexcept {
    ::vDSP_vmaD(A, IA, B, IB, C, IC, D, ID, N);

}
inline void vma(
                const SplitComplex<float> *A,
                Stride IA,
                const SplitComplex<float> *B,
                Stride IB,
                const SplitComplex<float> *C,
                Stride IC,
                const SplitComplex<float> *D,
                Stride ID,
                Length N) noexcept {
    ::vDSP_zvma(A, IA, B, IB, C, IC, D, ID, N);

}
inline void vma(
                const SplitComplex<double> *A,
                Stride IA,
                const SplitComplex<double> *B,
                Stride IB,
                const SplitComplex<double> *C,
                Stride IC,
                const SplitComplex<double> *D,
                Stride ID,
                Length N) noexcept {
    ::vDSP_zvmaD(A, IA, B, IB, C, IC, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = A[n] * B[n] + C[n];
 */


// Complex multiplication with optional conjugation.
inline void vmul(
                 const SplitComplex<float> *A,
                 Stride IA,
                 const SplitComplex<float> *B,
                 Stride IB,
                 const SplitComplex<float> *C,
                 Stride IC,
                 Length N,
                 int Conjugate) noexcept {
    ::vDSP_zvmul(A, IA, B, IB, C, IC, N, Conjugate);

}
inline void vmul(
                 const SplitComplex<double> *A,
                 Stride IA,
                 const SplitComplex<double> *B,
                 Stride IB,
                 const SplitComplex<double> *C,
                 Stride IC,
                 Length N,
                 int Conjugate) noexcept {
    ::vDSP_zvmulD(A, IA, B, IB, C, IC, N, Conjugate);

}
/*  Maps:  The default maps are used.

 These compute:

 If Conjugate is +1:

 for (n = 0; n < N; ++n)
 C[n] = A[n] * B[n];

 If Conjugate is -1:

 for (n = 0; n < N; ++n)
 C[n] = conj(A[n]) * B[n];
 */


// Complex-split inner (conjugate) dot product.
inline void zidotpr(
                    const SplitComplex<float> *A,
                    Stride IA,
                    const SplitComplex<float> *B,
                    Stride IB,
                    const SplitComplex<float> *C,
                    Length N) noexcept {
    ::vDSP_zidotpr(A, IA, B, IB, C, N);

}
inline void zidotprD(
                     const SplitComplex<double> *A,
                     Stride IA,
                     const SplitComplex<double> *B,
                     Stride IB,
                     const SplitComplex<double> *C,
                     Length N) noexcept {
    ::vDSP_zidotprD(A, IA, B, IB, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum(conj(A[n]) * B[n], 0 <= n < N);
 */


// Complex-split conjugate multiply and add.
inline void zvcma(
                  const SplitComplex<float> *A,
                  Stride IA,
                  const SplitComplex<float> *B,
                  Stride IB,
                  const SplitComplex<float> *C,
                  Stride IC,
                  const SplitComplex<float> *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_zvcma(A, IA, B, IB, C, IC, D, ID, N);

}
inline void zvcma(
                  const SplitComplex<double> *A,
                  Stride IA,
                  const SplitComplex<double> *B,
                  Stride IB,
                  const SplitComplex<double> *C,
                  Stride IC,
                  const SplitComplex<double> *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_zvcmaD(A, IA, B, IB, C, IC, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = conj(A[n]) * B[n] + C[n];
 */


// Subtract real from complex-split.
inline void vsub(
                 const SplitComplex<float> *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 const SplitComplex<float> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zrvsub(A, IA, B, IB, C, IC, N);

}
inline void vsub(
                 const SplitComplex<double> *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 const SplitComplex<double> *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zrvsubD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] - B[n];
 */


// Vector convert between double precision and single precision.
inline void vdpsp(
                  const double *A,
                  Stride IA,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vdpsp(A, IA, C, IC, N);

}
inline void vspdp(
                  const float *A,
                  Stride IA,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vspdp(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n];
 */


// Vector absolute value.
inline void vabs(
                 const float *A,
                 Stride IA,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vabs(A, IA, C, IC, N);

}
inline void vabs(
                 const double *A,
                 Stride IA,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vabsD(A, IA, C, IC, N);

}
inline void vabs(
                 const int *A,
                 Stride IA,
                 int *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vabsi(A, IA, C, IC, N);

}
inline void vabs(
                 const SplitComplex<float> *A,
                 Stride IA,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zvabs(A, IA, C, IC, N);

}
inline void vabs(
                 const SplitComplex<double> *A,
                 Stride IA,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_zvabsD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = |A[n]|;
 */


// Vector bit-wise equivalence, NOT (A XOR B).
inline void veqvi(
                  const int *A,
                  Stride IA,
                  const int *B,
                  Stride IB,
                  int *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_veqvi(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = ~(A[n] ^ B[n]);
 */


// Vector fill.
inline void vfill(
                  const float *A,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vfill(A, C, IC, N);

}
inline void vfill(
                  const double *A,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vfillD(A, C, IC, N);

}
inline void vfill(
                  const int *A,
                  int *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vfilli(A, C, IC, N);

}
inline void vfill(
                  const SplitComplex<float> *A,
                  const SplitComplex<float> *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_zvfill(A, C, IC, N);

}
inline void vfill(
                  const SplitComplex<double> *A,
                  const SplitComplex<double> *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_zvfillD(A, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[0];
 */


// Vector-scalar add.
inline void vsadd(
                  const float *A,
                  Stride IA,
                  const float *B,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vsadd(A, IA, B, C, IC, N);

}
inline void vsadd(
                  const double *A,
                  Stride IA,
                  const double *B,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vsaddD(A, IA, B, C, IC, N);

}
inline void vsadd(
                  const int *A,
                  Stride IA,
                  const int *B,
                  int *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vsaddi(A, IA, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] + B[0];
 */


// Vector-scalar divide.
inline void vsdiv(
                  const float *A,
                  Stride IA,
                  const float *B,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vsdiv(A, IA, B, C, IC, N);

}
inline void vsdiv(
                  const double *A,
                  Stride IA,
                  const double *B,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vsdivD(A, IA, B, C, IC, N);

}
inline void vsdiv(
                  const int *A,
                  Stride IA,
                  const int *B,
                  int *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vsdivi(A, IA, B, C, IC, N);

}


#pragma mark -

/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] / B[0];
 */


// Complex-split accumulating autospectrum.
inline void zaspec(
                   const SplitComplex<float> *A,
                   float *C,
                   Length N) noexcept {
    ::vDSP_zaspec(A, C, N);

}
inline void zaspec(
                   const SplitComplex<double> *A,
                   double *C,
                   Length N) noexcept {
    ::vDSP_zaspecD(A, C, N);

}
/*  Maps:

 No strides are used; arrays map directly to memory.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] += |A[n]| ** 2;
 */


// Create Blackman window.
inline void blkman_window(
                          float *C,
                          Length N,
                          int Flag) noexcept {
    ::vDSP_blkman_window(C, N, Flag);

}
inline void blkman_window(
                          double *C,
                          Length N,
                          int Flag) noexcept {
    ::vDSP_blkman_windowD(C, N, Flag);

}
/*  Maps:

 No strides are used; the array maps directly to memory.

 These compute:

 If Flag & vDSP_HALF_WINDOW:
 Length = (N+1)/2;
 Else
 Length = N;

 for (n = 0; n < Length; ++n)
 {
 angle = 2*pi*n/N;
 C[n] = .42 - .5 * cos(angle) + .08 * cos(2*angle);
 }
 */


// Coherence function.
inline void zcoher(
                   const float *A,
                   const float *B,
                   const SplitComplex<float> *C,
                   float *D,
                   Length N) noexcept {
    ::vDSP_zcoher(A, B, C, D, N);

}
inline void zcoher(
                   const double *A,
                   const double *B,
                   const SplitComplex<double> *C,
                   double *D,
                   Length N) noexcept {
    ::vDSP_zcoherD(A, B, C, D, N);

}
/*  Maps:

 No strides are used; arrays map directly to memory.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = |C[n]| ** 2 / (A[n] * B[n]);
 */


// Anti-aliasing down-sample with real filter.
inline void desamp(
                   const float *A,  // Input signal.
                   Stride I,  // Sampling interval.
                   const float *F,  // Filter.
                   float *C,  // Output.
                   Length N,  // Output length.
                   Length P) noexcept {  // Filter length.
    ::vDSP_desamp(A, I, F, C, N, P);
}
inline void desamp(
                   const double *A, // Input signal.
                   Stride I, // Sampling interval.
                   const double *F, // Filter.
                   double *C, // Output.
                   Length N, // Output length.
                   Length P) noexcept { // Filter length.
    ::vDSP_desampD(A, I, F, C, N, P);
}
inline void desamp(
                   const SplitComplex<float> *A,  // Input signal.
                   Stride I,  // Sampling interval.
                   const float *F,  // Filter.
                   const SplitComplex<float> *C,  // Output.
                   Length N,  // Output length.
                   Length P) noexcept {  // Filter length.
    ::vDSP_zrdesamp(A, I, F, C, N, P);
}
inline void desamp(
                   const SplitComplex<double> *A,    // Input signal.
                   Stride I,    // Sampling interval.
                   const double *F,    // Filter.
                   const SplitComplex<double> *C,    // Output.
                   Length N,    // Output length.
                   Length P) noexcept {    // Filter length.
    ::vDSP_zrdesampD(A, I, F, C, N, P);
}
/*  Maps:

 No strides are used; arrays map directly to memory.  I specifies
 a sampling interval.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = sum(A[n*I+p] * F[p], 0 <= p < P);
 */


// Transfer function, B/A.
inline void ztrans(
                   const float *A,
                   const SplitComplex<float> *B,
                   const SplitComplex<float> *C,
                   Length N) noexcept {
    ::vDSP_ztrans(A, B, C, N);

}
inline void ztrans(
                   const double *A,
                   const SplitComplex<double> *B,
                   const SplitComplex<double> *C,
                   Length N) noexcept {
    ::vDSP_ztransD(A, B, C, N);

}
/*  Maps:

 No strides are used; arrays map directly to memory.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = B[n] / A[n];
 */


// Accumulating cross-spectrum.
inline void zcspec(
                   const SplitComplex<float> *A,
                   const SplitComplex<float> *B,
                   const SplitComplex<float> *C,
                   Length N) noexcept {
    ::vDSP_zcspec(A, B, C, N);

}
inline void zcspec(
                   const SplitComplex<double> *A,
                   const SplitComplex<double> *B,
                   const SplitComplex<double> *C,
                   Length N) noexcept {
    ::vDSP_zcspecD(A, B, C, N);

}
/*  Maps:

 No strides are used; arrays map directly to memory.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] += conj(A[n]) * B[n];
 */


// Vector conjugate and multiply.
inline void zvcmul(
                   const SplitComplex<float> *A,
                   Stride IA,
                   const SplitComplex<float> *B,
                   Stride IB,
                   const SplitComplex<float> *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvcmul(A, IA, B, IB, C, IC, N);

}
inline void zvcmul(
                   const SplitComplex<double> *A,
                   Stride IA,
                   const SplitComplex<double> *B,
                   Stride IB,
                   const SplitComplex<double> *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvcmulD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = conj(A[n]) * B[n];
 */


// Vector conjugate.
inline void zvconj(
                   const SplitComplex<float> *A,
                   Stride IA,
                   const SplitComplex<float> *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvconj(A, IA, C, IC, N);

}
inline void zvconj(
                   const SplitComplex<double> *A,
                   Stride IA,
                   const SplitComplex<double> *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvconjD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = conj(A[n]);
 */


// Vector multiply with scalar.
inline void zvzsml(
                   const SplitComplex<float> *A,
                   Stride IA,
                   const SplitComplex<float> *B,
                   const SplitComplex<float> *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvzsml(A, IA, B, C, IC, N);

}
inline void zvzsml(
                   const SplitComplex<double> *A,
                   Stride IA,
                   const SplitComplex<double> *B,
                   const SplitComplex<double> *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvzsmlD(A, IA, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] * B[0];
 */


// Vector magnitudes squared.
inline void zvmags(
                   const SplitComplex<float> *A,
                   Stride IA,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvmags(A, IA, C, IC, N);

}
inline void zvmags(
                   const SplitComplex<double> *A,
                   Stride IA,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvmagsD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = |A[n]| ** 2;
 */


// Vector magnitudes square and add.
inline void zvmgsa(
                   const SplitComplex<float> *A,
                   Stride IA,
                   const float *B,
                   Stride IB,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvmgsa(A, IA, B, IB, C, IC, N);

}
inline void zvmgsa(
                   const SplitComplex<double> *A,
                   Stride IA,
                   const double *B,
                   Stride IB,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvmgsaD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = |A[n]| ** 2 + B[n];
 */


// Complex-split vector move.
inline void zvmov(
                  const SplitComplex<float> *A,
                  Stride IA,
                  const SplitComplex<float> *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_zvmov(A, IA, C, IC, N);

}
inline void zvmov(
                  const SplitComplex<double> *A,
                  Stride IA,
                  const SplitComplex<double> *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_zvmovD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n];
 */


// Vector negate.
inline void zvneg(
                  const SplitComplex<float> *A,
                  Stride IA,
                  const SplitComplex<float> *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_zvneg(A, IA, C, IC, N);

}
inline void zvneg(
                  const SplitComplex<double> *A,
                  Stride IA,
                  const SplitComplex<double> *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_zvnegD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = -A[n];
 */


// Vector phasea.
inline void zvphas(
                   const SplitComplex<float> *A,
                   Stride IA,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvphas(A, IA, C, IC, N);

}
inline void zvphas(
                   const SplitComplex<double> *A,
                   Stride IA,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_zvphasD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = atan2(Im(A[n]), Re(A[n]));
 */


// Vector multiply by scalar and add.
inline void zvsma(
                  const SplitComplex<float> *A,
                  Stride IA,
                  const SplitComplex<float> *B,
                  const SplitComplex<float> *C,
                  Stride IC,
                  const SplitComplex<float> *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_zvsma(A, IA, B, C, IC, D, ID, N);

}
inline void zvsma(
                  const SplitComplex<double> *A,
                  Stride IA,
                  const SplitComplex<double> *B,
                  const SplitComplex<double> *C,
                  Stride IC,
                  const SplitComplex<double> *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_zvsmaD(A, IA, B, C, IC, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = A[n] * B[0] + C[n];
 */


// Difference equation, 2 poles, 2 zeros.
inline void deq22(
                  const float *A,
                  Stride IA,
                  const float *B,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_deq22(A, IA, B, C, IC, N);

}
inline void deq22(
                  const double *A,
                  Stride IA,
                  const double *B,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_deq22D(A, IA, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 2; n < N+2; ++n)   // Note outputs start with C[2].
 C[n] =
 + A[n-0]*B[0]
 + A[n-1]*B[1]
 + A[n-2]*B[2]
 - C[n-1]*B[3]
 - C[n-2]*B[4];
 */


// Create Hamming window.
inline void hamm_window(
                        float *C,
                        Length N,
                        int Flag) noexcept {
    ::vDSP_hamm_window(C, N, Flag);

}
inline void hamm_window(
                        double *C,
                        Length N,
                        int Flag) noexcept {
    ::vDSP_hamm_windowD(C, N, Flag);

}
/*  Maps:

 No strides are used; the array maps directly to memory.

 These compute:

 If Flag & vDSP_HALF_WINDOW:
 Length = (N+1)/2;
 Else
 Length = N;

 for (n = 0; n < Length; ++n)
 C[n] = .54 - .46 * cos(2*pi*n/N);
 */


// Create Hanning window.
inline void hann_window(
                        float *C,
                        Length N,
                        int Flag) noexcept {
    ::vDSP_hann_window(C, N, Flag);

}
inline void hann_window(
                        double *C,
                        Length N,
                        int Flag) noexcept {
    ::vDSP_hann_windowD(C, N, Flag);

}
/*  Maps:

 No strides are used; the array maps directly to memory.

 These compute:

 If Flag & vDSP_HALF_WINDOW:
 Length = (N+1)/2;
 Else
 Length = N;

 If Flag & vDSP_HANN_NORM:
 W = .8165;
 Else
 W = .5;

 for (n = 0; n < Length; ++n)
 C[n] = W * (1 - cos(2*pi*n/N));
 */


// Maximum magnitude of vector.
inline void maxmgv(
                   const float *A,
                   Stride IA,
                   float *C,
                   Length N) noexcept {
    ::vDSP_maxmgv(A, IA, C, N);

}
inline void maxmgv(
                   const double *A,
                   Stride IA,
                   double *C,
                   Length N) noexcept {
    ::vDSP_maxmgvD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 C[0] is set to the greatest value of |A[n]| for 0 <= n < N.
 */


// Maximum magnitude of vector.
inline void maxmgvi(
                    const float *A,
                    Stride IA,
                    float *C,
                    Length *I,
                    Length N) noexcept {
    ::vDSP_maxmgvi(A, IA, C, I, N);

}
inline void maxmgvi(
                    const double *A,
                    Stride IA,
                    double *C,
                    Length *I,
                    Length N) noexcept {
    ::vDSP_maxmgviD(A, IA, C, I, N);

}
/*  Maps:  The default maps are used.

 C[0] is set to the greatest value of |A[n]| for 0 <= n < N.
 I[0] is set to the least i*IA such that |A[i]| has the value in C[0].
 */


// Maximum value of vector.
inline void maxv(
                 const float *A,
                 Stride IA,
                 float *C,
                 Length N) noexcept {
    ::vDSP_maxv(A, IA, C, N);

}
inline void maxv(
                 const double *A,
                 Stride IA,
                 double *C,
                 Length N) noexcept {
    ::vDSP_maxvD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 C[0] is set to the greatest value of A[n] for 0 <= n < N.
 */


// Maximum value of vector, with index.
inline void maxvi(
                  const float *A,
                  Stride IA,
                  float *C,
                  Length *I,
                  Length N) noexcept {
    ::vDSP_maxvi(A, IA, C, I, N);

}
inline void maxvi(
                  const double *A,
                  Stride IA,
                  double *C,
                  Length *I,
                  Length N) noexcept {
    ::vDSP_maxviD(A, IA, C, I, N);

}
/*  Maps:  The default maps are used.

 C[0] is set to the greatest value of A[n] for 0 <= n < N.
 I[0] is set to the least i*IA such that A[i] has the value in C[0].
 */


// Mean magnitude of vector.
inline void meamgv(
                   const float *A,
                   Stride IA,
                   float *C,
                   Length N) noexcept {
    ::vDSP_meamgv(A, IA, C, N);

}
inline void meamgv(
                   const double *A,
                   Stride IA,
                   double *C,
                   Length N) noexcept {
    ::vDSP_meamgvD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum(|A[n]|, 0 <= n < N) / N;
 */


// Mean of vector.
inline void meanv(
                  const float *A,
                  Stride IA,
                  float *C,
                  Length N) noexcept {
    ::vDSP_meanv(A, IA, C, N);

}
inline void meanv(
                  const double *A,
                  Stride IA,
                  double *C,
                  Length N) noexcept {
    ::vDSP_meanvD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum(A[n], 0 <= n < N) / N;
 */


// Mean square of vector.
inline void measqv(
                   const float *A,
                   Stride IA,
                   float *C,
                   Length N) noexcept {
    ::vDSP_measqv(A, IA, C, N);

}
inline void measqv(
                   const double *A,
                   Stride IA,
                   double *C,
                   Length N) noexcept {
    ::vDSP_measqvD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum(A[n]**2, 0 <= n < N) / N;
 */


// Minimum magnitude of vector.
inline void minmgv(
                   const float *A,
                   Stride IA,
                   float *C,
                   Length N) noexcept {
    ::vDSP_minmgv(A, IA, C, N);

}
inline void minmgv(
                   const double *A,
                   Stride IA,
                   double *C,
                   Length N) noexcept {
    ::vDSP_minmgvD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 C[0] is set to the least value of |A[n]| for 0 <= n < N.
 */


// Minimum magnitude of vector, with index.
inline void minmgvi(
                    const float *A,
                    Stride IA,
                    float *C,
                    Length *I,
                    Length N) noexcept {
    ::vDSP_minmgvi(A, IA, C, I, N);

}
inline void minmgvi(
                    const double *A,
                    Stride IA,
                    double *C,
                    Length *I,
                    Length N) noexcept {
    ::vDSP_minmgviD(A, IA, C, I, N);

}
/*  Maps:  The default maps are used.

 C[0] is set to the least value of |A[n]| for 0 <= n < N.
 I[0] is set to the least i*IA such that |A[i]| has the value in C[0].
 */


// Minimum value of vector.
inline void minv(
                 const float *A,
                 Stride IA,
                 float *C,
                 Length N) noexcept {
    ::vDSP_minv(A, IA, C, N);

}
inline void minv(
                 const double *A,
                 Stride IA,
                 double *C,
                 Length N) noexcept {
    ::vDSP_minvD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 C[0] is set to the least value of A[n] for 0 <= n < N.
 */


// Minimum value of vector, with index.
inline void minvi(
                  const float *A,
                  Stride IA,
                  float *C,
                  Length *I,
                  Length N) noexcept {
    ::vDSP_minvi(A, IA, C, I, N);

}
inline void minvi(
                  const double *A,
                  Stride IA,
                  double *C,
                  Length *I,
                  Length N) noexcept {
    ::vDSP_minviD(A, IA, C, I, N);

}
/*  Maps:  The default maps are used.

 C[0] is set to the least value of A[n] for 0 <= n < N.
 I[0] is set to the least i*IA such that A[i] has the value in C[0].
 */


// Matrix move.
inline void mmov(
                 const float *A,
                 float *C,
                 Length M,
                 Length N,
                 Length TA,
                 Length TC) noexcept {
    ::vDSP_mmov(A, C, M, N, TA, TC);

}
inline void mmov(
                 const double *A,
                 double *C,
                 Length M,
                 Length N,
                 Length TA,
                 Length TC) noexcept {
    ::vDSP_mmovD(A, C, M, N, TA, TC);

}
/*  Maps:

 This routine does not have strides.

 A is regarded as a two-dimensional matrix with dimensions [N][TA].
 C is regarded as a two-dimensional matrix with dimensions [N][TC].

 These compute:

 for (n = 0; n < N; ++n)
 for (m = 0; m < M; ++m)
 C[n][m] = A[n][m];
 */


// Mean of signed squares of vector.
inline void mvessq(
                   const float *A,
                   Stride IA,
                   float *C,
                   Length N) noexcept {
    ::vDSP_mvessq(A, IA, C, N);

}
inline void mvessq(
                   const double *A,
                   Stride IA,
                   double *C,
                   Length N) noexcept {
    ::vDSP_mvessqD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum(A[n] * |A[n]|, 0 <= n < N) / N;
 */


// Find zero crossing.
inline void nzcros(
                   const float *A,
                   Stride IA,
                   Length B,
                   Length *C,
                   Length *D,
                   Length N) noexcept {
    ::vDSP_nzcros(A, IA, B, C, D, N);

}
inline void nzcros(
                   const double *A,
                   Stride IA,
                   Length B,
                   Length *C,
                   Length *D,
                   Length N) noexcept {
    ::vDSP_nzcrosD(A, IA, B, C, D, N);

}
/*  Maps:  The default maps are used.

 Let S be the number of times the sign bit changes in the sequence A[0],
 A[1],... A[N-1].

 If B <= S:
 D[0] is set to B.
 C[0] is set to n*IA, where the B-th sign bit change occurs between
 elements A[n-1] and A[n].
 Else:
 D[0] is set to S.
 C[0] is set to 0.
 */


// Convert rectangular to polar.
inline void polar(
                  const float *A,
                  Stride IA,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_polar(A, IA, C, IC, N);

}
inline void polar(
                  const double *A,
                  Stride IA,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_polarD(A, IA, C, IC, N);

}
/*  Maps:  Strides are shown explicitly in pseudocode.

 These compute:

 for (n = 0; n < N; ++n)
 {
 x = A[n*IA+0];
 y = A[n*IA+1];
 C[n*IC+0] = sqrt(x**2 + y**2);
 C[n*IC+1] = atan2(y, x);
 }
 */


// Convert polar to rectangular.
inline void rect(
                 const float *A,
                 Stride IA,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_rect(A, IA, C, IC, N);

}
inline void rect(
                 const double *A,
                 Stride IA,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_rectD(A, IA, C, IC, N);

}
/*  Maps:  Strides are shown explicitly in pseudocode.

 These compute:

 for (n = 0; n < N; ++n)
 {
 r     = A[n*IA+0];
 theta = A[n*IA+1];
 C[n*IC+0] = r * cos(theta);
 C[n*IC+1] = r * sin(theta);
 }
 */


// Root-mean-square of vector.
inline void rmsqv(
                  const float *A,
                  Stride IA,
                  float *C,
                  Length N) noexcept {
    ::vDSP_rmsqv(A, IA, C, N);

}
inline void rmsqv(
                  const double *A,
                  Stride IA,
                  double *C,
                  Length N) noexcept {
    ::vDSP_rmsqvD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sqrt(sum(A[n] ** 2, 0 <= n < N) / N);
 */


// Scalar-vector divide.
inline void svdiv(
                  const float *A,
                  const float *B,
                  Stride IB,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_svdiv(A, B, IB, C, IC, N);

}
inline void svdiv(
                  const double *A,
                  const double *B,
                  Stride IB,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_svdivD(A, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[0] / B[n];

 When A[0] is not zero or NaN and B[n] is zero, C[n] is set to an
 infinity.
 */


// Sum of vector elements.
inline void sve(
                const float *A,
                Stride I,
                float *C,
                Length N) noexcept {
    ::vDSP_sve(A, I, C, N);

}
inline void sve(
                const double *A,
                Stride I,
                double *C,
                Length N) noexcept {
    ::vDSP_sveD(A, I, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum(A[n], 0 <= n < N);
 */


// Sum of vector elements magnitudes.
inline void svemg(
                  const float *A,
                  Stride IA,
                  float *C,
                  Length N) noexcept {
    ::vDSP_svemg(A, IA, C, N);

}
inline void svemg(
                  const double *A,
                  Stride IA,
                  double *C,
                  Length N) noexcept {
    ::vDSP_svemgD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum(|A[n]|, 0 <= n < N);
 */


// Sum of vector elements' squares.
inline void svesq(
                  const float *A,
                  Stride IA,
                  float *C,
                  Length N) noexcept {
    ::vDSP_svesq(A, IA, C, N);

}
inline void svesq(
                  const double *A,
                  Stride IA,
                  double *C,
                  Length N) noexcept {
    ::vDSP_svesqD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum(A[n] ** 2, 0 <= n < N);
 */


// Sum of vector elements and sum of vector elements' squares.
inline void sve_svesq(
                      const float *A,
                      Stride IA,
                      float *Sum,
                      float *SumOfSquares,
                      Length N) noexcept {
    ::vDSP_sve_svesq(A, IA, Sum, SumOfSquares, N);

}
inline void sve_svesq(
                      const double *A,
                      Stride IA,
                      double *Sum,
                      double *SumOfSquares,
                      Length N) noexcept {
    ::vDSP_sve_svesqD(A, IA, Sum, SumOfSquares, N);

}
/*  Maps:  The default maps are used.

 These compute:

 Sum[0]          = sum(A[n],      0 <= n < N);
 SumOfSquares[0] = sum(A[n] ** 2, 0 <= n < N);
 */


// Normalize elements to zero mean and unit standard deviation.
inline void normalize(
                      const float *A,
                      Stride IA,
                      float *C,
                      Stride IC,
                      float *Mean,
                      float *StandardDeviation,
                      Length N) noexcept {
    ::vDSP_normalize(A, IA, C, IC, Mean, StandardDeviation, N);

}
inline void normalize(
                      const double *A,
                      Stride IA,
                      double *C,
                      Stride IC,
                      double *Mean,
                      double *StandardDeviation,
                      Length N) noexcept {
    ::vDSP_normalizeD(A, IA, C, IC, Mean, StandardDeviation, N);

}
/*  Maps:  The default maps are used.

 These compute:

 // Calculate mean and standard deviation.
 m = sum(A[n], 0 <= n < N) / N;
 d = sqrt(sum(A[n]**2, 0 <= n < N) / N - m**2);

 // Normalize.
 for (n = 0; n < N; ++n)
 C[n] = (A[n] - m) / d;
 */


// Sum of vector elements' signed squares.
inline void svs(
                const float *A,
                Stride IA,
                float *C,
                Length N) noexcept {
    ::vDSP_svs(A, IA, C, N);

}
inline void svs(
                const double *A,
                Stride IA,
                double *C,
                Length N) noexcept {
    ::vDSP_svsD(A, IA, C, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = sum(A[n] * |A[n]|, 0 <= n < N);
 */


// Vector add, add, and multiply.
inline void vaam(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 const float *C,
                 Stride IC,
                 const float *D,
                 Stride ID,
                 float *E,
                 Stride IE,
                 Length N) noexcept {
    ::vDSP_vaam(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
inline void vaam(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 const double *C,
                 Stride IC,
                 const double *D,
                 Stride ID,
                 double *E,
                 Stride IE,
                 Length N) noexcept {
    ::vDSP_vaamD(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 E[n] = (A[n] + B[n]) * (C[n] + D[n]);
 */


// Vector add, subtract, and multiply.
inline void vasbm(
                  const float *A,
                  Stride IA,
                  const float *B,
                  Stride IB,
                  const float *C,
                  Stride IC,
                  const float *D,
                  Stride ID,
                  float *E,
                  Stride IE,
                  Length N) noexcept {
    ::vDSP_vasbm(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
inline void vasbm(
                  const double *A,
                  Stride IA,
                  const double *B,
                  Stride IB,
                  const double *C,
                  Stride IC,
                  const double *D,
                  Stride ID,
                  double *E,
                  Stride IE,
                  Length N) noexcept {
    ::vDSP_vasbmD(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 E[n] = (A[n] + B[n]) * (C[n] - D[n]);
 */


// Vector add and scalar multiply.
inline void vasm(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 const float *C,
                 float *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vasm(A, IA, B, IB, C, D, ID, N);

}
inline void vasm(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 const double *C,
                 double *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vasmD(A, IA, B, IB, C, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = (A[n] + B[n]) * C[0];
 */


// Vector linear average.
inline void vavlin(
                   const float *A,
                   Stride IA,
                   const float *B,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vavlin(A, IA, B, C, IC, N);

}
inline void vavlin(
                   const double *A,
                   Stride IA,
                   const double *B,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vavlinD(A, IA, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = (C[n]*B[0] + A[n]) / (B[0] + 1);
 */


// Vector clip.
inline void vclip(
                  const float *A,
                  Stride IA,
                  const float *B,
                  const float *C,
                  float *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_vclip(A, IA, B, C, D, ID, N);

}
inline void vclip(
                  const double *A,
                  Stride IA,
                  const double *B,
                  const double *C,
                  double *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_vclipD(A, IA, B, C, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 {
 D[n] = A[n];
 if (D[n] < B[0]) D[n] = B[0];
 if (C[0] < D[n]) D[n] = C[0];
 }
 */


// Vector clip and count.
inline void vclipc(
                   const float *A,
                   Stride IA,
                   const float *B,
                   const float *C,
                   float *D,
                   Stride ID,
                   Length N,
                   Length *NLow,
                   Length *NHigh) noexcept {
    ::vDSP_vclipc(A, IA, B, C, D, ID, N, NLow, NHigh);

}
inline void vclipc(
                   const double *A,
                   Stride IA,
                   const double *B,
                   const double *C,
                   double *D,
                   Stride ID,
                   Length N,
                   Length *NLow,
                   Length *NHigh) noexcept {
    ::vDSP_vclipcD(A, IA, B, C, D, ID, N, NLow, NHigh);

}
/*  Maps:  The default maps are used.

 These compute:

 NLow[0]  = 0;
 NHigh[0] = 0;
 for (n = 0; n < N; ++n)
 {
 D[n] = A[n];
 if (D[n] < B[0]) noexcept { D[n] = B[0]; ++NLow[0];  }
 if (C[0] < D[n]) noexcept { D[n] = C[0]; ++NHigh[0]; }
 }
 */


// Vector clear.
inline void vclr(
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vclr(C, IC, N);

}
inline void vclr(
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vclrD(C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = 0;
 */


// Vector compress.
inline void vcmprs(
                   const float *A,
                   Stride IA,
                   const float *B,
                   Stride IB,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vcmprs(A, IA, B, IB, C, IC, N);

}
inline void vcmprs(
                   const double *A,
                   Stride IA,
                   const double *B,
                   Stride IB,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vcmprsD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 p = 0;
 for (n = 0; n < N; ++n)
 if (B[n] != 0)
 C[p++] = A[n];
 */


// Vector convert to decibels, power, or amplitude.
inline void vdbcon(
                   const float *A,
                   Stride IA,
                   const float *B,
                   float *C,
                   Stride IC,
                   Length N,
                   unsigned int F) noexcept {
    ::vDSP_vdbcon(A, IA, B, C, IC, N, F);

}
inline void vdbcon(
                   const double *A,
                   Stride IA,
                   const double *B,
                   double *C,
                   Stride IC,
                   Length N,
                   unsigned int F) noexcept {
    ::vDSP_vdbconD(A, IA, B, C, IC, N, F);

}
/*  Maps:  The default maps are used.

 These compute:

 If Flag is 1:
 alpha = 20;
 If Flag is 0:
 alpha = 10;

 for (n = 0; n < N; ++n)
 C[n] = alpha * log10(A[n] / B[0]);
 */


// Vector distance.
inline void vdist(
                  const float *A,
                  Stride I,
                  const float *B,
                  Stride J,
                  float *C,
                  Stride K,
                  Length N) noexcept {
    ::vDSP_vdist(A, I, B, J, C, K, N);

}
inline void vdist(
                  const double *A,
                  Stride I,
                  const double *B,
                  Stride J,
                  double *C,
                  Stride K,
                  Length N) noexcept {
    ::vDSP_vdistD(A, I, B, J, C, K, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = sqrt(A[n]**2 + B[n]**2);
 */


// Vector envelope.
inline void venvlp(
                   const float *A,
                   Stride IA,
                   const float *B,
                   Stride IB,
                   const float *C,
                   Stride IC,
                   float *D,
                   Stride ID,
                   Length N) noexcept {
    ::vDSP_venvlp(A, IA, B, IB, C, IC, D, ID, N);

}
inline void venvlp(
                   const double *A,
                   Stride IA,
                   const double *B,
                   Stride IB,
                   const double *C,
                   Stride IC,
                   double *D,
                   Stride ID,
                   Length N) noexcept {
    ::vDSP_venvlpD(A, IA, B, IB, C, IC, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 {
 if (D[n] < B[n] || A[n] < D[n]) D[n] = C[n];
 else D[n] = 0;
 }
 */


// Vector convert to integer, round toward zero.
inline void vfix8(
                  const float *A,
                  Stride IA,
                  char *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vfix8(A, IA, C, IC, N);

}
inline void vfix8(
                  const double *A,
                  Stride IA,
                  char *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vfix8D(A, IA, C, IC, N);

}
inline void vfix16(
                   const float *A,
                   Stride IA,
                   short *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vfix16(A, IA, C, IC, N);

}
inline void vfix16(
                   const double *A,
                   Stride IA,
                   short *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vfix16D(A, IA, C, IC, N);

}
inline void vfix32(
                   const float *A,
                   Stride IA,
                   int *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vfix32(A, IA, C, IC, N);

}
inline void vfix32(
                   const double *A,
                   Stride IA,
                   int *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vfix32D(A, IA, C, IC, N);

}
inline void vfixu8(
                   const float *A,
                   Stride IA,
                   unsigned char *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vfixu8(A, IA, C, IC, N);

}
inline void vfixu8(
                   const double *A,
                   Stride IA,
                   unsigned char *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vfixu8D(A, IA, C, IC, N);

}
inline void vfixu16(
                    const float *A,
                    Stride IA,
                    unsigned short *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfixu16(A, IA, C, IC, N);

}
inline void vfixu16(
                    const double *A,
                    Stride IA,
                    unsigned short *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfixu16D(A, IA, C, IC, N);

}
inline void vfixu32(
                    const float *A,
                    Stride IA,
                    unsigned int *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfixu32(A, IA, C, IC, N);

}
inline void vfixu32(
                    const double *A,
                    Stride IA,
                    unsigned int *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfixu32D(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = trunc(A[n]);
 */

// Vector convert single precision to 24 bit integer with pre-scaling.
// The scaled value is rounded toward zero.
inline void vsmfixu24(
                      const float *A,
                      Stride IA,
                      const float *B,
                      uint24 *C,
                      Stride IC,
                      Length N) noexcept {
    ::vDSP_vsmfixu24(A, IA, B, C, IC, N);

}

// Vector convert single precision to 24 bit unsigned integer with pre-scaling.
// The scaled value is rounded toward zero.
inline void vsmfix24(
                     const float *A,
                     Stride IA,
                     const float *B,
                     int24  *C,
                     Stride IC,
                     Length N) noexcept {
    ::vDSP_vsmfix24(A, IA, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = trunc(A[n] * B[0]);

 Note: Values outside the representable range are clamped to the largest or smallest
 representable values of the destination type.
 */

// Vector convert unsigned 24 bit integer to single precision float
inline void vfltu24(
                    const uint24 *A,
                    Stride IA,
                    float *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfltu24(A, IA, C, IC, N);

}

// Vector convert 24 bit integer to single precision float
inline void vflt24(
                   const int24 *A,
                   Stride IA,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vflt24(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n];
 */

// Vector convert unsigned 24 bit integer to single precision float and scale
inline void vfltsmu24(
                      const uint24 *A,
                      Stride IA,
                      const float *B,
                      float *C,
                      Stride IC,
                      Length N) noexcept {
    ::vDSP_vfltsmu24(A, IA, B, C, IC, N);

}

// Vector convert 24 bit integer to single precision float and scale
inline void vfltsm24(
                     const int24 *A,
                     Stride IA,
                     const float *B,
                     float *C,
                     Stride IC,
                     Length N) noexcept {
    ::vDSP_vfltsm24(A, IA, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = B[0] * (float)A[n];
 */

// Vector convert to integer, round to nearest.
inline void vfixr8(
                   const float *A,
                   Stride IA,
                   char *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vfixr8(A, IA, C, IC, N);

}
inline void vfixr8(
                   const double *A,
                   Stride IA,
                   char *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vfixr8D(A, IA, C, IC, N);

}
inline void vfixr16(
                    const float *A,
                    Stride IA,
                    short *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfixr16(A, IA, C, IC, N);

}
inline void vfixr16(
                    const double *A,
                    Stride IA,
                    short *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfixr16D(A, IA, C, IC, N);

}
inline void vfixr32(
                    const float *A,
                    Stride IA,
                    int *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfixr32(A, IA, C, IC, N);

}
inline void vfixr32(
                    const double *A,
                    Stride IA,
                    int *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfixr32D(A, IA, C, IC, N);

}
inline void vfixru8(
                    const float *A,
                    Stride IA,
                    unsigned char *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfixru8(A, IA, C, IC, N);

}
inline void vfixru8(
                    const double *A,
                    Stride IA,
                    unsigned char *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfixru8D(A, IA, C, IC, N);

}
inline void vfixru16(
                     const float *A,
                     Stride IA,
                     unsigned short *C,
                     Stride IC,
                     Length N) noexcept {
    ::vDSP_vfixru16(A, IA, C, IC, N);

}
inline void vfixru16(
                     const double *A,
                     Stride IA,
                     unsigned short *C,
                     Stride IC,
                     Length N) noexcept {
    ::vDSP_vfixru16D(A, IA, C, IC, N);

}
inline void vfixru32(
                     const float *A,
                     Stride IA,
                     unsigned int *C,
                     Stride IC,
                     Length N) noexcept {
    ::vDSP_vfixru32(A, IA, C, IC, N);

}
inline void vfixru32(
                     const double *A,
                     Stride IA,
                     unsigned int *C,
                     Stride IC,
                     Length N) noexcept {
    ::vDSP_vfixru32D(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = rint(A[n]);

 Note:  It is expected that the global rounding mode be the default,
 round-to-nearest.  It is unspecified whether ties round up or down.
 */


// Vector convert to floating-point from integer.
inline void vflt8(
                  const char *A,
                  Stride IA,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vflt8(A, IA, C, IC, N);

}
inline void vflt8(
                  const char *A,
                  Stride IA,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vflt8D(A, IA, C, IC, N);

}
inline void vflt16(
                   const short *A,
                   Stride IA,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vflt16(A, IA, C, IC, N);

}
inline void vflt16(
                   const short *A,
                   Stride IA,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vflt16D(A, IA, C, IC, N);

}
inline void vflt32(
                   const int *A,
                   Stride IA,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vflt32(A, IA, C, IC, N);

}
inline void vflt32(
                   const int *A,
                   Stride IA,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vflt32D(A, IA, C, IC, N);

}
inline void vfltu8(
                   const unsigned char *A,
                   Stride IA,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vfltu8(A, IA, C, IC, N);

}
inline void vfltu8(
                   const unsigned char *A,
                   Stride IA,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vfltu8D(A, IA, C, IC, N);

}
inline void vfltu16(
                    const unsigned short *A,
                    Stride IA,
                    float *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfltu16(A, IA, C, IC, N);

}
inline void vfltu16(
                    const unsigned short *A,
                    Stride IA,
                    double *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfltu16D(A, IA, C, IC, N);

}
inline void vfltu32(
                    const unsigned int *A,
                    Stride IA,
                    float *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfltu32(A, IA, C, IC, N);

}
inline void vfltu32(
                    const unsigned int *A,
                    Stride IA,
                    double *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vfltu32D(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n];
 */


// Vector fraction part (subtract integer toward zero).
inline void vfrac(
                  const float *A,
                  Stride IA,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vfrac(A, IA, C, IC, N);

}
inline void vfrac(
                  const double *A,
                  Stride IA,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vfracD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] - trunc(A[n]);
 */


// Vector gather.
inline void vgathr(
                   const float *A,
                   const Length *B,
                   Stride IB,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vgathr(A, B, IB, C, IC, N);

}
inline void vgathr(
                   const double *A,
                   const Length *B,
                   Stride IB,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vgathrD(A, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.  Note that A has unit stride.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[B[n] - 1];
 */


// Vector gather, absolute pointers.
inline void vgathra(
                    const float **A,
                    Stride IA,
                    float *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vgathra(A, IA, C, IC, N);

}
inline void vgathra(
                    const double **A,
                    Stride IA,
                    double *C,
                    Stride IC,
                    Length N) noexcept {
    ::vDSP_vgathraD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = *A[n];
 */


// Vector generate tapered ramp.
inline void vgen(
                 const float *A,
                 const float *B,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vgen(A, B, C, IC, N);

}
inline void vgen(
                 const double *A,
                 const double *B,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vgenD(A, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[0] + (B[0] - A[0]) * n/(N-1);
 */


// Vector generate by extrapolation and interpolation.
inline void vgenp(
                  const float *A,
                  Stride IA,
                  const float *B,
                  Stride IB,
                  float *C,
                  Stride IC,
                  Length N,
                  Length M) noexcept {  // Length of A and of B.
    ::vDSP_vgenp(A, IA, B, IB, C, IC, N, M);
}
inline void vgenp(
                  const double *A,
                  Stride IA,
                  const double *B,
                  Stride IB,
                  double *C,
                  Stride IC,
                  Length N,
                  Length M) noexcept { // Length of A and of B.
    ::vDSP_vgenpD(A, IA, B, IB, C, IC, N, M);
}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 If n <= B[0],  then C[n] = A[0].
 If B[M-1] < n, then C[n] = A[M-1].
 Otherwise:
 Let m be such that B[m] < n <= B[m+1].
 C[n] = A[m] + (A[m+1]-A[m]) * (n-B[m]) / (B[m+1]-B[m]).

 The elements of B are expected to be in increasing order.
 */


// Vector inverted clip.
inline void viclip(
                   const float *A,
                   Stride IA,
                   const float *B,
                   const float *C,
                   float *D,
                   Stride ID,
                   Length N) noexcept {
    ::vDSP_viclip(A, IA, B, C, D, ID, N);

}
inline void viclip(
                   const double *A,
                   Stride IA,
                   const double *B,
                   const double *C,
                   double *D,
                   Stride ID,
                   Length N) noexcept {
    ::vDSP_viclipD(A, IA, B, C, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 {
 if (A[n] <= B[0] || C[0] <= A[n])
 D[n] = A[n];
 else
 if (A[n] < 0)
 D[n] = B[0];
 else
 D[n] = C[0];
 }
 */


// Vector index, C[i] = A[truncate[B[i]].
inline void vindex(
                   const float *A,
                   const float *B,
                   Stride IB,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vindex(A, B, IB, C, IC, N);
}
inline void vindex(
                   const double *A,
                   const double *B,
                   Stride IB,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vindexD(A, B, IB, C, IC, N);
}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[trunc(B[n])];
 */


// Vector interpolation between vectors.
inline void vintb(
                  const float *A,
                  Stride IA,
                  const float *B,
                  Stride IB,
                  const float *C,
                  float *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_vintb(A, IA, B, IB, C, D, ID, N);
}
inline void vintb(
                  const double *A,
                  Stride IA,
                  const double *B,
                  Stride IB,
                  const double *C,
                  double *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_vintbD(A, IA, B, IB, C, D, ID, N);
}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = A[n] + C[0] * (B[n] - A[n]);
 */


// Vector test limit.
inline void vlim(
                 const float *A,
                 Stride IA,
                 const float *B,
                 const float *C,
                 float *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vlim(A, IA, B, C, D, ID, N);
}
inline void vlim(
                 const double *A,
                 Stride IA,
                 const double *B,
                 const double *C,
                 double *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vlimD(A, IA, B, C, D, ID, N);
}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 if (B[0] <= A[n])
 D[n] = +C[0];
 else
 D[n] = -C[0];
 */


// Vector linear interpolation.
inline void vlint(
                  const float *A,
                  const float *B,
                  Stride IB,
                  float *C,
                  Stride IC,
                  Length N,
                  Length M) noexcept {  // Nominal length of A, but not used.
    ::vDSP_vlint(A, B, IB, C, IC, N, M);
}
inline void vlint(
                  const double *A,
                  const double *B,
                  Stride IB,
                  double *C,
                  Stride IC,
                  Length N,
                  Length M) noexcept { // Nominal length of A, but not used.
    ::vDSP_vlintD(A, B, IB, C, IC, N, M);
}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 {
 b = trunc(B[n]);
 a = B[n] - b;
 C[n] = A[b] + a * (A[b+1] - A[b]);
 }
 */


// Vector maxima.
inline void vmax(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vmax(A, IA, B, IB, C, IC, N);

}
inline void vmax(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vmaxD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = B[n] <= A[n] ? A[n] : B[n];
 */


// Vector maximum magnitude.
inline void vmaxmg(
                   const float *A,
                   Stride IA,
                   const float *B,
                   Stride IB,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vmaxmg(A, IA, B, IB, C, IC, N);

}
inline void vmaxmg(
                   const double *A,
                   Stride IA,
                   const double *B,
                   Stride IB,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vmaxmgD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = |B[n]| <= |A[n]| ? |A[n]| : |B[n]|;
 */


// Vector sliding window maxima.
inline void vswmax(
                   const float *A,
                   Stride IA,
                   float *C,
                   Stride IC,
                   Length N,
                   Length WindowLength) noexcept {
    ::vDSP_vswmax(A, IA, C, IC, N, WindowLength);
}
inline void vswmax(
                   const double *A,
                   Stride IA,
                   double *C,
                   Stride IC,
                   Length N,
                   Length WindowLength) noexcept {
    ::vDSP_vswmaxD(A, IA, C, IC, N, WindowLength);
}
/*  Maps:  The default maps are used.

 These compute the maximum value within a window to the input vector.
 A maximum is calculated for each window position:

 for (n = 0; n < N; ++n)
 C[n] = the greatest value of A[w] for n <= w < n+WindowLength.

 A must contain N+WindowLength-1 elements, and C must contain space for
 N+WindowLength-1 elements.  Although only N outputs are provided in C,
 the additional elements may be used for intermediate computation.

 A and C may not overlap.

 WindowLength must be positive (zero is not supported).
 */


// Vector minima.
inline void vmin(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vmin(A, IA, B, IB, C, IC, N);

}
inline void vmin(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vminD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] <= B[n] ? A[n] : B[n];
 */


// Vector minimum magnitude.
inline void vminmg(
                   const float *A,
                   Stride IA,
                   const float *B,
                   Stride IB,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vminmg(A, IA, B, IB, C, IC, N);

}
inline void vminmg(
                   const double *A,
                   Stride IA,
                   const double *B,
                   Stride IB,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vminmgD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = |A[n]| <= |B[n]| ? |A[n]| : |B[n]|;
 */


// Vector multiply, multiply, and add.
inline void vmma(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 const float *C,
                 Stride IC,
                 const float *D,
                 Stride ID,
                 float *E,
                 Stride IE,
                 Length N) noexcept {
    ::vDSP_vmma(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
inline void vmma(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 const double *C,
                 Stride IC,
                 const double *D,
                 Stride ID,
                 double *E,
                 Stride IE,
                 Length N) noexcept {
    ::vDSP_vmmaD(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 E[n] = A[n]*B[n] + C[n]*D[n];
 */


// Vector multiply, multiply, and subtract.
inline void vmmsb(
                  const float *A,
                  Stride IA,
                  const float *B,
                  Stride IB,
                  const float *C,
                  Stride IC,
                  const float *D,
                  Stride ID,
                  float *E,
                  Stride IE,
                  Length N) noexcept {
    ::vDSP_vmmsb(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
inline void vmmsb(
                  const double *A,
                  Stride IA,
                  const double *B,
                  Stride IB,
                  const double *C,
                  Stride IC,
                  const double *D,
                  Stride ID,
                  double *E,
                  Stride IE,
                  Length N) noexcept {
    ::vDSP_vmmsbD(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 E[n] = A[n]*B[n] - C[n]*D[n];
 */


// Vector multiply and scalar add.
inline void vmsa(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 const float *C,
                 float *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vmsa(A, IA, B, IB, C, D, ID, N);

}
inline void vmsa(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 const double *C,
                 double *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vmsaD(A, IA, B, IB, C, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = A[n]*B[n] + C[0];
 */


// Vector multiply and subtract.
inline void vmsb(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 const float *C,
                 Stride IC,
                 float *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vmsb(A, IA, B, IB, C, IC, D, ID, N);

}
inline void vmsb(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 const double *C,
                 Stride IC,
                 double *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vmsbD(A, IA, B, IB, C, IC, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = A[n]*B[n] - C[n];
 */


// Vector negative absolute value.
inline void vnabs(
                  const float *A,
                  Stride IA,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vnabs(A, IA, C, IC, N);

}
inline void vnabs(
                  const double *A,
                  Stride IA,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vnabsD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = -|A[n]|;
 */


// Vector negate.
inline void vneg(
                 const float *A,
                 Stride IA,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vneg(A, IA, C, IC, N);

}
inline void vneg(
                 const double *A,
                 Stride IA,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vnegD(A, IA, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = -A[n];
 */


// Vector polynomial.
inline void vpoly(
                  const float *A,
                  Stride IA,
                  const float *B,
                  Stride IB,
                  float *C,
                  Stride IC,
                  Length N,
                  Length P) noexcept { // P is the polynomial degree.
    ::vDSP_vpoly(A, IA, B, IB, C, IC, N, P);
}
inline void vpoly(
                  const double *A,
                  Stride IA,
                  const double *B,
                  Stride IB,
                  double *C,
                  Stride IC,
                  Length N,
                  Length P) noexcept {  // P is the polynomial degree.
    ::vDSP_vpolyD(A, IA, B, IB, C, IC, N, P);
}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = sum(A[P-p] * B[n]**p, 0 <= p <= P);
 */


// Vector Pythagoras.
inline void vpythg(
                   const float *A,
                   Stride IA,
                   const float *B,
                   Stride IB,
                   const float *C,
                   Stride IC,
                   const float *D,
                   Stride ID,
                   float *E,
                   Stride IE,
                   Length N) noexcept {
    ::vDSP_vpythg(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
inline void vpythg(
                   const double *A,
                   Stride IA,
                   const double *B,
                   Stride IB,
                   const double *C,
                   Stride IC,
                   const double *D,
                   Stride ID,
                   double *E,
                   Stride IE,
                   Length N) noexcept {
    ::vDSP_vpythgD(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 E[n] = sqrt((A[n]-C[n])**2 + (B[n]-D[n])**2);
 */


// Vector quadratic interpolation.
inline void vqint(
                  const float *A,
                  const float *B,
                  Stride IB,
                  float *C,
                  Stride IC,
                  Length N,
                  Length M) noexcept {
    ::vDSP_vqint(A, B, IB, C, IC, N, M);

}
inline void vqint(
                  const double *A,
                  const double *B,
                  Stride IB,
                  double *C,
                  Stride IC,
                  Length N,
                  Length M) noexcept {
    ::vDSP_vqintD(A, B, IB, C, IC, N, M);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 {
 b = max(trunc(B[n]), 1);
 a = B[n] - b;
 C[n] = (A[b-1]*(a**2-a) + A[b]*(2-2*a**2) + A[b+1]*(a**2+a))
 / 2;
 }
 */


// Vector build ramp.
inline void vramp(
                  const float *A,
                  const float *B,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vramp(A, B, C, IC, N);

}
inline void vramp(
                  const double *A,
                  const double *B,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vrampD(A, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[0] + n*B[0];
 */


// Vector running sum integration.
inline void vrsum(
                  const float *A,
                  Stride IA,
                  const float *S,
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vrsum(A, IA, S, C, IC, N);

}
inline void vrsum(
                  const double *A,
                  Stride IA,
                  const double *S,
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vrsumD(A, IA, S, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = S[0] * sum(A[j], 0 < j <= n);

 Observe that C[0] is set to 0, and A[0] is not used.
 */


// Vector reverse order, in-place.
inline void vrvrs(
                  float *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vrvrs(C, IC, N);

}
inline void vrvrs(
                  double *C,
                  Stride IC,
                  Length N) noexcept {
    ::vDSP_vrvrsD(C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 Let A contain a copy of C.
 for (n = 0; n < N; ++n)
 C[n] = A[N-1-n];
 */


// Vector subtract and multiply.
inline void vsbm(
                 const float *A,
                 Stride IA,
                 const float *B,
                 Stride IB,
                 const float *C,
                 Stride IC,
                 float *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vsbm(A, IA, B, IB, C, IC, D, ID, N);

}
inline void vsbm(
                 const double *A,
                 Stride IA,
                 const double *B,
                 Stride IB,
                 const double *C,
                 Stride IC,
                 double *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vsbmD(A, IA, B, IB, C, IC, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = (A[n] - B[n]) * C[n];
 */


// Vector subtract, subtract, and multiply.
inline void vsbsbm(
                   const float *A,
                   Stride IA,
                   const float *B,
                   Stride IB,
                   const float *C,
                   Stride IC,
                   const float *D,
                   Stride ID,
                   float *E,
                   Stride IE,
                   Length N) noexcept {
    ::vDSP_vsbsbm(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
inline void vsbsbm(
                   const double *A,
                   Stride IA,
                   const double *B,
                   Stride IB,
                   const double *C,
                   Stride IC,
                   const double *D,
                   Stride ID,
                   double *E,
                   Stride IE,
                   Length N) noexcept {
    ::vDSP_vsbsbmD(A, IA, B, IB, C, IC, D, ID, E, IE, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = (A[n] - B[n]) * (C[n] - D[n]);
 */


// Vector subtract and scalar multiply.
inline void vsbsm(
                  const float *A,
                  Stride IA,
                  const float *B,
                  Stride IB,
                  const float *C,
                  float *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_vsbsm(A, IA, B, IB, C, D, ID, N);

}
inline void vsbsm(
                  const double *A,
                  Stride IA,
                  const double *B,
                  Stride IB,
                  const double *C,
                  double *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_vsbsmD(A, IA, B, IB, C, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = (A[n] - B[n]) * C[0];
 */


// Vector Simpson integration.
inline void vsimps(
                   const float *A,
                   Stride IA,
                   const float *B,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vsimps(A, IA, B, C, IC, N);

}
inline void vsimps(
                   const double *A,
                   Stride IA,
                   const double *B,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vsimpsD(A, IA, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 C[0] = 0;
 C[1] = B[0] * (A[0] + A[1])/2;
 for (n = 2; n < N; ++n)
 C[n] = C[n-2] + B[0] * (A[n-2] + 4*A[n-1] + A[n])/3;
 */


// Vector-scalar multiply and vector add.
inline void vsma(
                 const float *A,
                 Stride IA,
                 const float *B,
                 const float *C,
                 Stride IC,
                 float *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vsma(A, IA, B, C, IC, D, ID, N);

}
inline void vsma(
                 const double *A,
                 Stride IA,
                 const double *B,
                 const double *C,
                 Stride IC,
                 double *D,
                 Stride ID,
                 Length N) noexcept {
    ::vDSP_vsmaD(A, IA, B, C, IC, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = A[n]*B[0] + C[n];
 */


// Vector-scalar multiply and scalar add.
inline void vsmsa(
                  const float *A,
                  Stride IA,
                  const float *B,
                  const float *C,
                  float *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_vsmsa(A, IA, B, C, D, ID, N);

}
inline void vsmsa(
                  const double *A,
                  Stride IA,
                  const double *B,
                  const double *C,
                  double *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_vsmsaD(A, IA, B, C, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = A[n]*B[0] + C[0];
 */


// Vector scalar multiply and vector subtract.
inline void vsmsb(
                  const float *A,
                  Stride I,
                  const float *B,
                  const float *C,
                  Stride K,
                  float *D,
                  Stride L,
                  Length N) noexcept {
    ::vDSP_vsmsb(A, I, B, C, K, D, L, N);

}
inline void vsmsb(
                  const double *A,
                  Stride I,
                  const double *B,
                  const double *C,
                  Stride K,
                  double *D,
                  Stride L,
                  Length N) noexcept {
    ::vDSP_vsmsbD(A, I, B, C, K, D, L, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 D[n] = A[n]*B[0] - C[n];
 */


// Vector-scalar multiply, vector-scalar multiply and vector add.
inline void vsmsma(
                   const float *A,
                   Stride IA,
                   const float *B,
                   const float *C,
                   Stride IC,
                   const float *D,
                   float *E,
                   Stride IE,
                   Length N) noexcept {
    ::vDSP_vsmsma(A, IA, B, C, IC, D, E, IE, N);

}
inline void vsmsma(
                   const double *A,
                   Stride IA,
                   const double *B,
                   const double *C,
                   Stride IC,
                   const double *D,
                   double *E,
                   Stride IE,
                   Length N) noexcept {
    ::vDSP_vsmsmaD(A, IA, B, C, IC, D, E, IE, N);

}
/*  Maps:  The default maps are used.

 This computes:

 for (n = 0; n < N; ++n)
 E[n] = A[n]*B[0] + C[n]*D[0];
 */


// Vector sort, in-place.
inline void vsort(
                  float *C,
                  Length N,
                  int Order) noexcept {
    ::vDSP_vsort(C, N, Order);

}
inline void vsort(
                  double *C,
                  Length N,
                  int Order) noexcept {
    ::vDSP_vsortD(C, N, Order);

}
/*  If Order is +1, C is sorted in ascending order.
 If Order is -1, C is sorted in descending order.
 */


// Vector sort indices, in-place.
inline void vsorti(
                   const float *C,
                   Length *I,
                   Length *Temporary,
                   Length N,
                   int Order) noexcept {
    ::vDSP_vsorti(C, I, Temporary, N, Order);

}
inline void vsorti(
                   const double *C,
                   Length *I,
                   Length *Temporary,
                   Length N,
                   int Order) noexcept {
    ::vDSP_vsortiD(C, I, Temporary, N, Order);

}
/*  Maps:  No strides are used; arrays map directly to memory.

 I contains indices into C.

 If Order is +1, I is sorted so that C[I[n]] increases, for 0 <= n < N.
 If Order is -1, I is sorted so that C[I[n]] decreases, for 0 <= n < N.

 Temporary is not used.  NULL should be passed for it.
 */


// Vector swap.
inline void vswap(
                  float *A,
                  Stride IA,
                  float *B,
                  Stride IB,
                  Length N) noexcept {
    ::vDSP_vswap(A, IA, B, IB, N);

}
inline void vswap(
                  double *A,
                  Stride IA,
                  double *B,
                  Stride IB,
                  Length N) noexcept {
    ::vDSP_vswapD(A, IA, B, IB, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 A[n] is swapped with B[n].
 */


// Vector sliding window sum.
inline void vswsum(
                   const float *A,
                   Stride IA,
                   float *C,
                   Stride IC,
                   Length N,
                   Length P) noexcept { // Length of window.
    ::vDSP_vswsum(A, IA, C, IC, N, P);
}
inline void vswsum(
                   const double *A,
                   Stride IA,
                   double *C,
                   Stride IC,
                   Length N,
                   Length P) noexcept { // Length of window.
    ::vDSP_vswsumD(A, IA, C, IC, N, P);
}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = sum(A[n+p], 0 <= p < P);

 Note that A must contain N+P-1 elements.
 */


// Vector table lookup and interpolation.
inline void vtabi(
                  const float *A,
                  Stride IA,
                  const float *S1,
                  const float *S2,
                  const float *C,
                  Length M,
                  float *D,
                  Stride ID,
                  Length N) noexcept {
    ::vDSP_vtabi(A, IA, S1, S2, C, M, D, ID, N);

}
inline void vtabi(
                  const double *A,
                  Stride IA,
                  const double *S1,
                  const double *S2,
                  const double *C,
                  Length M,
                  double *D,
                  Stride L,
                  Length N) noexcept {
    ::vDSP_vtabiD(A, IA, S1, S2, C, M, D, L, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 {
 p = S1[0] * A[n] + S2[0];
 if (p < 0)
 D[n] = C[0];
 else if (p < M-1)
 {
 q = trunc(p);
 r = p-q;
 D[n] = (1-r)*C[q] + r*C[q+1];
 }
 else
 D[n] = C[M-1];
 }
 */


// Vector threshold.
inline void vthr(
                 const float *A,
                 Stride IA,
                 const float *B,
                 float *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vthr(A, IA, B, C, IC, N);

}
inline void vthr(
                 const double *A,
                 Stride IA,
                 const double *B,
                 double *C,
                 Stride IC,
                 Length N) noexcept {
    ::vDSP_vthrD(A, IA, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 if (B[0] <= A[n])
 C[n] = A[n];
 else
 C[n] = B[0];
 */


// Vector threshold with zero fill.
inline void vthres(
                   const float *A,
                   Stride IA,
                   const float *B,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vthres(A, IA, B, C, IC, N);

}
inline void vthres(
                   const double *A,
                   Stride IA,
                   const double *B,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vthresD(A, IA, B, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 if (B[0] <= A[n])
 C[n] = A[n];
 else
 C[n] = 0;
 */


// Vector threshold with signed constant.
inline void vthrsc(
                   const float *A,
                   Stride IA,
                   const float *B,
                   const float *C,
                   float *D,
                   Stride ID,
                   Length N) noexcept {
    ::vDSP_vthrsc(A, IA, B, C, D, ID, N);

}
inline void vthrsc(
                   const double *A,
                   Stride IA,
                   const double *B,
                   const double *C,
                   double *D,
                   Stride ID,
                   Length N) noexcept {
    ::vDSP_vthrscD(A, IA, B, C, D, ID, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 if (B[0] <= A[n])
 D[n] = +C[0];
 else
 D[n] = -C[0];
 */


// Vector tapered merge.
inline void vtmerg(
                   const float *A,
                   Stride IA,
                   const float *B,
                   Stride IB,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vtmerg(A, IA, B, IB, C, IC, N);

}
inline void vtmerg(
                   const double *A,
                   Stride IA,
                   const double *B,
                   Stride IB,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vtmergD(A, IA, B, IB, C, IC, N);

}
/*  Maps:  The default maps are used.

 These compute:

 for (n = 0; n < N; ++n)
 C[n] = A[n] + (B[n] - A[n]) * n/(N-1);
 */


// Vector trapezoidal integration.
inline void vtrapz(
                   const float *A,
                   Stride IA,
                   const float *B,
                   float *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vtrapz(A, IA, B, C, IC, N);
    
}
inline void vtrapz(
                   const double *A,
                   Stride IA,
                   const double *B,
                   double *C,
                   Stride IC,
                   Length N) noexcept {
    ::vDSP_vtrapzD(A, IA, B, C, IC, N);
    
}
/*  Maps:  The default maps are used.
 
 These compute:
 
 C[0] = 0;
 for (n = 1; n < N; ++n)
 C[n] = C[n-1] + B[0] * (A[n-1] + A[n])/2;
 */


// Wiener Levinson.
inline void wiener(
                   Length L,
                   const float *A,
                   const float *C,
                   float *F,
                   float *P,
                   int Flag,
                   int *Error) noexcept {
    ::vDSP_wiener(L, A, C, F, P, Flag, Error);
    
}
inline void wiener(
                   Length L,
                   const double *A,
                   const double *C,
                   double *F,
                   double *P,
                   int Flag,
                   int *Error) noexcept {
    ::vDSP_wienerD(L, A, C, F, P, Flag, Error);
    
}

} // namespace dsp
} // namespace caffe
