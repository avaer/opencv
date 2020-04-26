/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*M///////////////////////////////////////////////////////////////////////////////////////
// Author: Sajjad Taheri, University of California, Irvine. sajjadt[at]uci[dot]edu
//
//                             LICENSE AGREEMENT
// Copyright (c) 2015 The Regents of the University of California (Regents)
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the University nor the
//    names of its contributors may be used to endorse or promote products
//    derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//M*/

#include <emscripten/emscripten.h>
#include <emscripten/bind.h>

@INCLUDES@
#include "../../../modules/core/src/parallel_impl.hpp"

#ifdef TEST_WASM_INTRIN
#include "../../../modules/core/include/opencv2/core/hal/intrin.hpp"
#include "../../../modules/core/include/opencv2/core/utils/trace.hpp"
#include "../../../modules/ts/include/opencv2/ts/ts_gtest.h"
namespace cv {
namespace hal {
#include "../../../modules/core/test/test_intrin_utils.hpp"
}
}
#endif

using namespace emscripten;
using namespace cv;

#ifdef HAVE_OPENCV_DNN
using namespace dnn;
#endif

#ifdef HAVE_OPENCV_ARUCO
// using namespace aruco;
#endif

#include <iostream>

// #include "../../../modules/core/include/opencv2/core/core.hpp"
#include "../../../modules/imgcodecs/include/opencv2/imgcodecs.hpp"
#include "../../../modules/imgproc/include/opencv2/imgproc.hpp"
#include "../../../modules/objdetect/include/opencv2/objdetect.hpp"
#include "../../../modules/features2d/include/opencv2/features2d.hpp"
#include "../../../opencv_contrib/modules/xfeatures2d/include/opencv2/xfeatures2d.hpp"
#include "../../../opencv_contrib/modules/xfeatures2d/include/opencv2/xfeatures2d/nonfree.hpp"

namespace binding_utils
{
    template<typename classT, typename enumT>
    static inline typename std::underlying_type<enumT>::type classT::* underlying_ptr(enumT classT::* enum_ptr)
    {
        return reinterpret_cast<typename std::underlying_type<enumT>::type classT::*>(enum_ptr);
    }

    template<typename T>
    emscripten::val matData(const cv::Mat& mat)
    {
        return emscripten::val(emscripten::memory_view<T>((mat.total()*mat.elemSize())/sizeof(T),
                               (T*)mat.data));
    }

    template<typename T>
    emscripten::val matPtr(const cv::Mat& mat, int i)
    {
        return emscripten::val(emscripten::memory_view<T>(mat.step1(0), mat.ptr<T>(i)));
    }

    template<typename T>
    emscripten::val matPtr(const cv::Mat& mat, int i, int j)
    {
        return emscripten::val(emscripten::memory_view<T>(mat.step1(1), mat.ptr<T>(i,j)));
    }

    cv::Mat* createMat(int rows, int cols, int type, intptr_t data, size_t step)
    {
        return new cv::Mat(rows, cols, type, reinterpret_cast<void*>(data), step);
    }

    static emscripten::val getMatSize(const cv::Mat& mat)
    {
        emscripten::val size = emscripten::val::array();
        for (int i = 0; i < mat.dims; i++) {
            size.call<void>("push", mat.size[i]);
        }
        return size;
    }

    static emscripten::val getMatStep(const cv::Mat& mat)
    {
        emscripten::val step = emscripten::val::array();
        for (int i = 0; i < mat.dims; i++) {
            step.call<void>("push", mat.step[i]);
        }
        return step;
    }

    static Mat matEye(int rows, int cols, int type)
    {
        return Mat(cv::Mat::eye(rows, cols, type));
    }

    static Mat matEye(Size size, int type)
    {
        return Mat(cv::Mat::eye(size, type));
    }

    void convertTo(const Mat& obj, Mat& m, int rtype, double alpha, double beta)
    {
        obj.convertTo(m, rtype, alpha, beta);
    }

    void convertTo(const Mat& obj, Mat& m, int rtype)
    {
        obj.convertTo(m, rtype);
    }

    void convertTo(const Mat& obj, Mat& m, int rtype, double alpha)
    {
        obj.convertTo(m, rtype, alpha);
    }

    Size matSize(const cv::Mat& mat)
    {
        return mat.size();
    }

    cv::Mat matZeros(int arg0, int arg1, int arg2)
    {
        return cv::Mat::zeros(arg0, arg1, arg2);
    }

    cv::Mat matZeros(cv::Size arg0, int arg1)
    {
        return cv::Mat::zeros(arg0,arg1);
    }

    cv::Mat matOnes(int arg0, int arg1, int arg2)
    {
        return cv::Mat::ones(arg0, arg1, arg2);
    }

    cv::Mat matOnes(cv::Size arg0, int arg1)
    {
        return cv::Mat::ones(arg0, arg1);
    }

    double matDot(const cv::Mat& obj, const Mat& mat)
    {
        return  obj.dot(mat);
    }

    Mat matMul(const cv::Mat& obj, const Mat& mat, double scale)
    {
        return  Mat(obj.mul(mat, scale));
    }

    Mat matT(const cv::Mat& obj)
    {
        return  Mat(obj.t());
    }

    Mat matInv(const cv::Mat& obj, int type)
    {
        return  Mat(obj.inv(type));
    }

    void matCopyTo(const cv::Mat& obj, cv::Mat& mat)
    {
        return obj.copyTo(mat);
    }

    void matCopyTo(const cv::Mat& obj, cv::Mat& mat, const cv::Mat& mask)
    {
        return obj.copyTo(mat, mask);
    }

    Mat matDiag(const cv::Mat& obj, int d)
    {
        return obj.diag(d);
    }

    Mat matDiag(const cv::Mat& obj)
    {
        return obj.diag();
    }

    void matSetTo(cv::Mat& obj, const cv::Scalar& s)
    {
        obj.setTo(s);
    }

    void matSetTo(cv::Mat& obj, const cv::Scalar& s, const cv::Mat& mask)
    {
        obj.setTo(s, mask);
    }

    emscripten::val rotatedRectPoints(const cv::RotatedRect& obj)
    {
        cv::Point2f points[4];
        obj.points(points);
        emscripten::val pointsArray = emscripten::val::array();
        for (int i = 0; i < 4; i++) {
            pointsArray.call<void>("push", points[i]);
        }
        return pointsArray;
    }

    Rect rotatedRectBoundingRect(const cv::RotatedRect& obj)
    {
        return obj.boundingRect();
    }

    Rect2f rotatedRectBoundingRect2f(const cv::RotatedRect& obj)
    {
        return obj.boundingRect2f();
    }

    int cvMatDepth(int flags)
    {
        return CV_MAT_DEPTH(flags);
    }

    class MinMaxLoc
    {
    public:
        double minVal;
        double maxVal;
        Point minLoc;
        Point maxLoc;
    };

    MinMaxLoc minMaxLoc(const cv::Mat& src, const cv::Mat& mask)
    {
        MinMaxLoc result;
        cv::minMaxLoc(src, &result.minVal, &result.maxVal, &result.minLoc, &result.maxLoc, mask);
        return result;
    }

    MinMaxLoc minMaxLoc_1(const cv::Mat& src)
    {
        MinMaxLoc result;
        cv::minMaxLoc(src, &result.minVal, &result.maxVal, &result.minLoc, &result.maxLoc);
        return result;
    }

    class Circle
    {
    public:
        Point2f center;
        float radius;
    };

#ifdef HAVE_OPENCV_IMGPROC
    Circle minEnclosingCircle(const cv::Mat& points)
    {
        Circle circle;
        cv::minEnclosingCircle(points, circle.center, circle.radius);
        return circle;
    }

    int floodFill_withRect_helper(cv::Mat& arg1, cv::Mat& arg2, Point arg3, Scalar arg4, emscripten::val arg5, Scalar arg6 = Scalar(), Scalar arg7 = Scalar(), int arg8 = 4)
    {
        cv::Rect rect;

        int rc = cv::floodFill(arg1, arg2, arg3, arg4, &rect, arg6, arg7, arg8);

        arg5.set("x", emscripten::val(rect.x));
        arg5.set("y", emscripten::val(rect.y));
        arg5.set("width", emscripten::val(rect.width));
        arg5.set("height", emscripten::val(rect.height));

        return rc;
    }

    int floodFill_wrapper(cv::Mat& arg1, cv::Mat& arg2, Point arg3, Scalar arg4, emscripten::val arg5, Scalar arg6, Scalar arg7, int arg8) {
        return floodFill_withRect_helper(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8);
    }

    int floodFill_wrapper_1(cv::Mat& arg1, cv::Mat& arg2, Point arg3, Scalar arg4, emscripten::val arg5, Scalar arg6, Scalar arg7) {
        return floodFill_withRect_helper(arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }

    int floodFill_wrapper_2(cv::Mat& arg1, cv::Mat& arg2, Point arg3, Scalar arg4, emscripten::val arg5, Scalar arg6) {
        return floodFill_withRect_helper(arg1, arg2, arg3, arg4, arg5, arg6);
    }

    int floodFill_wrapper_3(cv::Mat& arg1, cv::Mat& arg2, Point arg3, Scalar arg4, emscripten::val arg5) {
        return floodFill_withRect_helper(arg1, arg2, arg3, arg4, arg5);
    }

    int floodFill_wrapper_4(cv::Mat& arg1, cv::Mat& arg2, Point arg3, Scalar arg4) {
        return cv::floodFill(arg1, arg2, arg3, arg4);
    }
#endif

#ifdef HAVE_OPENCV_VIDEO
    emscripten::val CamShiftWrapper(const cv::Mat& arg1, Rect& arg2, TermCriteria arg3)
    {
        RotatedRect rotatedRect = cv::CamShift(arg1, arg2, arg3);
        emscripten::val result = emscripten::val::array();
        result.call<void>("push", rotatedRect);
        result.call<void>("push", arg2);
        return result;
    }

    emscripten::val meanShiftWrapper(const cv::Mat& arg1, Rect& arg2, TermCriteria arg3)
    {
        int n = cv::meanShift(arg1, arg2, arg3);
        emscripten::val result = emscripten::val::array();
        result.call<void>("push", n);
        result.call<void>("push", arg2);
        return result;
    }
#endif  // HAVE_OPENCV_VIDEO

    std::string getExceptionMsg(const cv::Exception& e) {
        return e.msg;
    }

    void setExceptionMsg(cv::Exception& e, std::string msg) {
        e.msg = msg;
        return;
    }

    cv::Exception exceptionFromPtr(intptr_t ptr) {
        return *reinterpret_cast<cv::Exception*>(ptr);
    }

    std::string getBuildInformation() {
        return cv::getBuildInformation();
    }

#ifdef TEST_WASM_INTRIN
    void test_hal_intrin_uint8() {
        cv::hal::test_hal_intrin_uint8();
    }
    void test_hal_intrin_int8() {
        cv::hal::test_hal_intrin_int8();
    }
    void test_hal_intrin_uint16() {
        cv::hal::test_hal_intrin_uint16();
    }
    void test_hal_intrin_int16() {
        cv::hal::test_hal_intrin_int16();
    }
    void test_hal_intrin_uint32() {
        cv::hal::test_hal_intrin_uint32();
    }
    void test_hal_intrin_int32() {
        cv::hal::test_hal_intrin_int32();
    }
    void test_hal_intrin_uint64() {
        cv::hal::test_hal_intrin_uint64();
    }
    void test_hal_intrin_int64() {
        cv::hal::test_hal_intrin_int64();
    }
    void test_hal_intrin_float32() {
        cv::hal::test_hal_intrin_float32();
    }
    void test_hal_intrin_float64() {
        cv::hal::test_hal_intrin_float64();
    }
    void test_hal_intrin_all() {
        cv::hal::test_hal_intrin_uint8();
        cv::hal::test_hal_intrin_int8();
        cv::hal::test_hal_intrin_uint16();
        cv::hal::test_hal_intrin_int16();
        cv::hal::test_hal_intrin_uint32();
        cv::hal::test_hal_intrin_int32();
        cv::hal::test_hal_intrin_uint64();
        cv::hal::test_hal_intrin_int64();
        cv::hal::test_hal_intrin_float32();
        cv::hal::test_hal_intrin_float64();
    }
#endif
}

float vectorLength(float x, float y, float z) {
  return std::sqrt(x*x + y*y + z*z);
}
/* void setPoseMatrix(float *dstMatrixArray, const vr::HmdMatrix44_t &srcMatrix) {
  for (unsigned int v = 0; v < 4; v++) {
    for (unsigned int u = 0; u < 4; u++) {
      dstMatrixArray[v * 4 + u] = srcMatrix.m[u][v];
    }
  }
}
void setPoseMatrix(float *dstMatrixArray, const vr::HmdMatrix34_t &srcMatrix) {
  for (unsigned int v = 0; v < 4; v++) {
    for (unsigned int u = 0; u < 3; u++) {
      dstMatrixArray[v * 4 + u] = srcMatrix.m[u][v];
    }
  }
  dstMatrixArray[0 * 4 + 3] = 0;
  dstMatrixArray[1 * 4 + 3] = 0;
  dstMatrixArray[2 * 4 + 3] = 0;
  dstMatrixArray[3 * 4 + 3] = 1;
}
void setPoseMatrix(vr::HmdMatrix34_t &dstMatrix, const float *srcMatrixArray) {
  for (unsigned int v = 0; v < 4; v++) {
    for (unsigned int u = 0; u < 3; u++) {
      dstMatrix.m[u][v] = srcMatrixArray[v * 4 + u];
    }
  }
} */
float matrixDeterminant(const float *matrix) {
  const float *te = matrix;

  float n11 = te[ 0 ], n12 = te[ 4 ], n13 = te[ 8 ], n14 = te[ 12 ];
  float n21 = te[ 1 ], n22 = te[ 5 ], n23 = te[ 9 ], n24 = te[ 13 ];
  float n31 = te[ 2 ], n32 = te[ 6 ], n33 = te[ 10 ], n34 = te[ 14 ];
  float n41 = te[ 3 ], n42 = te[ 7 ], n43 = te[ 11 ], n44 = te[ 15 ];

  //TODO: make this more efficient
  //( based on http://www.euclideanspace.com/maths/algebra/matrix/functions/inverse/fourD/index.htm )

  return (
    n41 * (
      + n14 * n23 * n32
       - n13 * n24 * n32
       - n14 * n22 * n33
       + n12 * n24 * n33
       + n13 * n22 * n34
       - n12 * n23 * n34
    ) +
    n42 * (
      + n11 * n23 * n34
       - n11 * n24 * n33
       + n14 * n21 * n33
       - n13 * n21 * n34
       + n13 * n24 * n31
       - n14 * n23 * n31
    ) +
    n43 * (
      + n11 * n24 * n32
       - n11 * n22 * n34
       - n14 * n21 * n32
       + n12 * n21 * n34
       + n14 * n22 * n31
       - n12 * n24 * n31
    ) +
    n44 * (
      - n13 * n22 * n31
       - n11 * n23 * n32
       + n11 * n22 * n33
       + n13 * n21 * n32
       - n12 * n21 * n33
       + n12 * n23 * n31
    )
  );
}
void getQuaternionFromRotationMatrix(float *quaternion, const float *matrix) {
  // http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
  // assumes the upper 3x3 of m is a pure rotation matrix (i.e, unscaled)

  const float *te = matrix;

  const float m11 = te[ 0 ], m12 = te[ 4 ], m13 = te[ 8 ],
    m21 = te[ 1 ], m22 = te[ 5 ], m23 = te[ 9 ],
    m31 = te[ 2 ], m32 = te[ 6 ], m33 = te[ 10 ];

  float trace = m11 + m22 + m33,
    s;

  if (trace > 0.0f) {
    s = 0.5f / std::sqrt(trace + 1.0f);

    quaternion[0] = (m32 - m23) * s;
    quaternion[1] = (m13 - m31) * s;
    quaternion[2] = (m21 - m12) * s;
    quaternion[3] = 0.25f / s;
  } else if (m11 > m22 && m11 > m33) {
    s = 2.0f * std::sqrt(1.0f + m11 - m22 - m33 );

    quaternion[0] = 0.25f * s;
    quaternion[1] = (m12 + m21) / s;
    quaternion[2] = (m13 + m31) / s;
    quaternion[3] = (m32 - m23) / s;
  } else if (m22 > m33) {
    s = 2.0f * std::sqrt(1.0f + m22 - m11 - m33);

    quaternion[0] = (m12 + m21) / s;
    quaternion[1] = 0.25f * s;
    quaternion[2] = (m23 + m32) / s;
    quaternion[3] = (m13 - m31) / s;
  } else {
    s = 2.0f * std::sqrt(1.0f + m33 - m11 - m22);

    quaternion[0] = (m13 + m31) / s;
    quaternion[1] = (m23 + m32) / s;
    quaternion[2] = 0.25f * s;
    quaternion[3] = (m21 - m12) / s;
  }
}
void getMatrixInverse(const float *inMatrix, float *outMatrix) {
  // based on http://www.euclideanspace.com/maths/algebra/matrix/functions/inverse/fourD/index.htm
  float *te = outMatrix;
  const float *me = inMatrix;

  const float n11 = me[ 0 ], n21 = me[ 1 ], n31 = me[ 2 ], n41 = me[ 3 ],
    n12 = me[ 4 ], n22 = me[ 5 ], n32 = me[ 6 ], n42 = me[ 7 ],
    n13 = me[ 8 ], n23 = me[ 9 ], n33 = me[ 10 ], n43 = me[ 11 ],
    n14 = me[ 12 ], n24 = me[ 13 ], n34 = me[ 14 ], n44 = me[ 15 ],

    t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44,
    t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44,
    t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44,
    t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

  const float det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;

  if (det == 0.0f) {
    std::cerr << "Can't invert matrix, determinant is 0" << std::endl;
  }

  const float detInv = 1.0f / det;

  te[ 0 ] = t11 * detInv;
  te[ 1 ] = ( n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44 ) * detInv;
  te[ 2 ] = ( n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44 ) * detInv;
  te[ 3 ] = ( n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43 ) * detInv;

  te[ 4 ] = t12 * detInv;
  te[ 5 ] = ( n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44 ) * detInv;
  te[ 6 ] = ( n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44 ) * detInv;
  te[ 7 ] = ( n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43 ) * detInv;

  te[ 8 ] = t13 * detInv;
  te[ 9 ] = ( n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44 ) * detInv;
  te[ 10 ] = ( n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44 ) * detInv;
  te[ 11 ] = ( n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43 ) * detInv;

  te[ 12 ] = t14 * detInv;
  te[ 13 ] = ( n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34 ) * detInv;
  te[ 14 ] = ( n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34 ) * detInv;
  te[ 15 ] = ( n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33 ) * detInv;
}
void multiplyMatrices(const float *aMatrix, const float *bMatrix, float *outMatrix) {
  const float *ae = aMatrix;
  const float *be = bMatrix;
  float *te = outMatrix;

  const float a11 = ae[ 0 ], a12 = ae[ 4 ], a13 = ae[ 8 ], a14 = ae[ 12 ];
  const float a21 = ae[ 1 ], a22 = ae[ 5 ], a23 = ae[ 9 ], a24 = ae[ 13 ];
  const float a31 = ae[ 2 ], a32 = ae[ 6 ], a33 = ae[ 10 ], a34 = ae[ 14 ];
  const float a41 = ae[ 3 ], a42 = ae[ 7 ], a43 = ae[ 11 ], a44 = ae[ 15 ];

  const float b11 = be[ 0 ], b12 = be[ 4 ], b13 = be[ 8 ], b14 = be[ 12 ];
  const float b21 = be[ 1 ], b22 = be[ 5 ], b23 = be[ 9 ], b24 = be[ 13 ];
  const float b31 = be[ 2 ], b32 = be[ 6 ], b33 = be[ 10 ], b34 = be[ 14 ];
  const float b41 = be[ 3 ], b42 = be[ 7 ], b43 = be[ 11 ], b44 = be[ 15 ];

  te[ 0 ] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
  te[ 4 ] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
  te[ 8 ] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
  te[ 12 ] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;

  te[ 1 ] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
  te[ 5 ] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
  te[ 9 ] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
  te[ 13 ] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;

  te[ 2 ] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
  te[ 6 ] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
  te[ 10 ] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
  te[ 14 ] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;

  te[ 3 ] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
  te[ 7 ] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
  te[ 11 ] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
  te[ 15 ] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;
}
void composeMatrix(float *matrix, const float *position, const float *quaternion, const float *scale) {
  float *te = matrix;

  float x = quaternion[0], y = quaternion[1], z = quaternion[2], w = quaternion[3];
  float x2 = x + x, y2 = y + y, z2 = z + z;
  float xx = x * x2, xy = x * y2, xz = x * z2;
  float yy = y * y2, yz = y * z2, zz = z * z2;
  float wx = w * x2, wy = w * y2, wz = w * z2;

  float sx = scale[0], sy = scale[1], sz = scale[2];

  te[ 0 ] = ( 1 - ( yy + zz ) ) * sx;
  te[ 1 ] = ( xy + wz ) * sx;
  te[ 2 ] = ( xz - wy ) * sx;
  te[ 3 ] = 0;

  te[ 4 ] = ( xy - wz ) * sy;
  te[ 5 ] = ( 1 - ( xx + zz ) ) * sy;
  te[ 6 ] = ( yz + wx ) * sy;
  te[ 7 ] = 0;

  te[ 8 ] = ( xz + wy ) * sz;
  te[ 9 ] = ( yz - wx ) * sz;
  te[ 10 ] = ( 1 - ( xx + yy ) ) * sz;
  te[ 11 ] = 0;

  te[ 12 ] = position[0];
  te[ 13 ] = position[1];
  te[ 14 ] = position[2];
  te[ 15 ] = 1;
}
void decomposeMatrix(const float *matrix, float *position, float *quaternion, float *scale) {
  const float *te = matrix;

  float sx = vectorLength(te[ 0 ], te[ 1 ], te[ 2 ]);
  float sy = vectorLength(te[ 4 ], te[ 5 ], te[ 6 ]);
  float sz = vectorLength(te[ 8 ], te[ 9 ], te[ 10 ]);

  // if determine is negative, we need to invert one scale
  float det = matrixDeterminant(matrix);
  if ( det < 0 ) sx = - sx;

  position[0] = te[ 12 ];
  position[1] = te[ 13 ];
  position[2] = te[ 14 ];

  // scale the rotation part
  float _m1[16];
  memcpy(_m1, matrix, 16 * sizeof(float));

  float invSX = 1.0f / sx;
  float invSY = 1.0f / sy;
  float invSZ = 1.0f / sz;

  _m1[ 0 ] *= invSX;
  _m1[ 1 ] *= invSX;
  _m1[ 2 ] *= invSX;

  _m1[ 4 ] *= invSY;
  _m1[ 5 ] *= invSY;
  _m1[ 6 ] *= invSY;

  _m1[ 8 ] *= invSZ;
  _m1[ 9 ] *= invSZ;
  _m1[ 10 ] *= invSZ;

  getQuaternionFromRotationMatrix(quaternion, matrix);

  scale[0] = sx;
  scale[1] = sy;
  scale[2] = sz;
}
void addVector3(float *a, const float *b) {
  a[0] += b[0];
  a[1] += b[1];
  a[2] += b[2];
}
void addVector4(float *a, const float *b) {
  a[0] += b[0];
  a[1] += b[1];
  a[2] += b[2];
  a[3] += b[3];
}
void applyVector3Quaternion(float *v, const float *q) {
  float x = v[0], y = v[1], z = v[2];
  float qx = q[0], qy = q[1], qz = q[2], qw = q[3];

  // calculate quat * vector

  float ix = qw * x + qy * z - qz * y;
  float iy = qw * y + qz * x - qx * z;
  float iz = qw * z + qx * y - qy * x;
  float iw = - qx * x - qy * y - qz * z;

  // calculate result * inverse quat

  v[0] = ix * qw + iw * - qx + iy * - qz - iz * - qy;
  v[1] = iy * qw + iw * - qy + iz * - qx - ix * - qz;
  v[2] = iz * qw + iw * - qz + ix * - qy - iy * - qx;
}
void applyVector3Matrix(float *v, const float *m) {
  float x = v[0], y = v[1], z = v[2];
  const float *e = m;

  float w = 1.0f / ( e[ 3 ] * x + e[ 7 ] * y + e[ 11 ] * z + e[ 15 ] );

  v[0] = ( e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z + e[ 12 ] ) * w;
  v[1] = ( e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z + e[ 13 ] ) * w;
  v[2] = ( e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z + e[ 14 ] ) * w;
}
void applyVector4Matrix(float *v, const float *m) {
  float x = v[0], y = v[1], z = v[2], w = v[3];
  const float *e = m;

  v[0] = e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z + e[ 12 ] * w;
  v[1] = e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z + e[ 13 ] * w;
  v[2] = e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z + e[ 14 ] * w;
  v[3] = e[ 3 ] * x + e[ 7 ] * y + e[ 11 ] * z + e[ 15 ] * w;
}
void multiplyVectors3(float *a, const float *b) {
  a[0] *= b[0];
  a[1] *= b[1];
  a[2] *= b[2];
}
void multiplyVectors4(float *a, const float *b) {
  a[0] *= b[0];
  a[1] *= b[1];
  a[2] *= b[2];
  a[3] *= b[3];
}
void divideVectors3(float *a, const float *b) {
  a[0] /= b[0];
  a[1] /= b[1];
  a[2] /= b[2];
}
void divideVectors4(float *a, const float *b) {
  a[0] /= b[0];
  a[1] /= b[1];
  a[2] /= b[2];
  a[3] /= b[3];
}
void multiplyVector3Scalar(float *v, const float s) {
  v[0] *= s;
  v[1] *= s;
  v[2] *= s;
}
void multiplyVector4Scalar(float *v, const float s) {
  v[0] *= s;
  v[1] *= s;
  v[2] *= s;
  v[3] *= s;
}
void addVector3Scalar(float *v, const float s) {
  v[0] += s;
  v[1] += s;
  v[2] += s;
}
void addVector4Scalar(float *v, const float s) {
  v[0] += s;
  v[1] += s;
  v[2] += s;
}
void perspectiveDivideVector(float *v) {
  v[0] /= v[3];
  v[1] /= v[3];
  v[2] /= v[3];
  v[3] = 1.0f;
}

void getQr(int width, int height, unsigned char *imgData, float *viewMatrixInverse, float *projectionMatrixInverse, float *qrCodes, unsigned int &numQrCodes) {
    cv::QRCodeDetector qrDecoder;

    cv::Mat inputImage(height, width, CV_8UC4, imgData);

    cv::Mat bbox, rectifiedImage;

    std::string data = qrDecoder.detectAndDecode(inputImage, bbox, rectifiedImage);
    std::cerr << "thread 9 " << data.length() << std::endl;

    numQrCodes = 0;

    if (data.length() > 0 && bbox.type() == CV_32FC2 && bbox.rows == 4 && bbox.cols == 1) {
      std::cerr << "Decoded QR code: " << data << " " <<
        bbox.at<cv::Point2f>(0).x << " " << bbox.at<cv::Point2f>(0).y << " " <<
        bbox.at<cv::Point2f>(1).x << " " << bbox.at<cv::Point2f>(1).y << " " <<
        bbox.at<cv::Point2f>(2).x << " " << bbox.at<cv::Point2f>(2).y << " " <<
        bbox.at<cv::Point2f>(3).x << " " << bbox.at<cv::Point2f>(3).y << " " <<
        std::endl;
      
      for (int i = 0; i < 4; i++) {
        const cv::Point2f &p = bbox.at<cv::Point2f>(i);
        float worldPoint[4] = {
          (p.x/(float)width) * 2.0f - 1.0f,
          (1.0f-(p.y/(float)height)) * 2.0f - 1.0f,
          0.0f,
          1.0f,
        };
        applyVector4Matrix(worldPoint, projectionMatrixInverse);
        perspectiveDivideVector(worldPoint);
        applyVector4Matrix(worldPoint, viewMatrixInverse);
        // applyVector4Matrix(worldPoint, stageMatrixInverse);

        qrCodes[i*3] = worldPoint[0];
        qrCodes[i*3+1] = worldPoint[1];
        qrCodes[i*3+2] = worldPoint[2];
      }
      numQrCodes = 1;
    }

  // getOut() << "thread 10 " << data.length() << std::endl;
  // running = false;
}

extern "C" {
EMSCRIPTEN_KEEPALIVE void *doMalloc(uint32_t size) {
  return malloc(size);
}
EMSCRIPTEN_KEEPALIVE void doFree(void *p) {
  free(p);
}
EMSCRIPTEN_KEEPALIVE void doComputeCvFeatures(int imageRows, int imageCols, int imageType, uint8_t *imageData, uint32_t imageDataSize, float **queryPoints, uint32_t *queryPointsSize, int *descriptorRows, int *descriptorCols, int *descriptorType, uint8_t **descriptorData, uint32_t *descriptorDataSize, int minHessian) {
  try {
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    
    cv::Mat inputImage(imageRows, imageCols, imageType);
    if (imageDataSize > 0) {
      memcpy(inputImage.data, imageData, imageDataSize);
    }

    // cv::Mat inputImage2;
    // cv::cvtColor(inputImage, inputImage2, cv::COLOR_RGBA2GRAY);
    // cv::Mat inputImage3;
    // cv::resize(inputImage2, inputImage3, cv::Size(512, (float)512 * (float)inputImage2.rows / (float)inputImage2.cols), 0, 0, cv::INTER_CUBIC);

    std::cout << "cv 1 " << imageRows << " " << imageCols << " " << imageType << " " << imageDataSize << std::endl;

    std::vector<cv::KeyPoint> queryKeypoints;
    cv::Mat queryDescriptors;
    
    std::cout << "cv 2" << std::endl;

    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
    // detector->setExtended(true);
    std::cout << "cv 3.1" << std::endl;
    detector->detectAndCompute(inputImage, cv::noArray(), queryKeypoints, queryDescriptors);
    std::cout << "cv 3.2" << std::endl; 

    {
      *queryPoints = (float *)malloc(queryKeypoints.size() * 2 * sizeof(float));
      for (size_t i = 0; i < queryKeypoints.size(); i++) {
        const cv::KeyPoint &keypoint = queryKeypoints[i];
        (*queryPoints)[i*2] = keypoint.pt.x;
        (*queryPoints)[i*2+1] = keypoint.pt.y;
      }
      *queryPointsSize = queryKeypoints.size() * 2 * sizeof(float);

      *descriptorRows = queryDescriptors.rows;
      std::cout << "cv 8.1" << std::endl;
      *descriptorCols = queryDescriptors.cols;
      std::cout << "cv 8.2" << std::endl;
      *descriptorType = queryDescriptors.type();
      std::cout << "cv 8.3" << std::endl;
      *descriptorData = (uint8_t *)malloc(queryDescriptors.total() * queryDescriptors.elemSize());
      if (queryDescriptors.total() * queryDescriptors.elemSize() > 0) {
        memcpy(*descriptorData, queryDescriptors.data, queryDescriptors.total() * queryDescriptors.elemSize());
      }
      std::cout << "cv 8.4 " << (void *)descriptorData << " " << (void *)descriptorDataSize << " " << queryDescriptors.total() << " " << queryDescriptors.elemSize()<< std::endl;
      *descriptorDataSize = queryDescriptors.total() * queryDescriptors.elemSize();
      std::cout << "cv 8.5" << std::endl;
    }
    
    std::cout << "cv 9" << std::endl;
  } catch(cv::Exception& e) {
    std::cout << "exception caught: " << e.what() << std::endl;
  }
  std::cout << "cv 10" << std::endl;
}
EMSCRIPTEN_KEEPALIVE void doMatchCvFeatures(int queryRows, int queryCols, int queryType, uint8_t *queryData, uint32_t queryDataSize, int trainRows, int trainCols, int trainType, uint8_t *trainData, uint32_t trainDataSize, uint32_t **matchIndices, uint32_t *matchIndicesSize, float ratio_thresh) {
  try {
    cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5); // instantiate LSH index parameters
    cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);       // instantiate flann search parameters
    std::unique_ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher(indexParams, searchParams));
    // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    std::cout << "cv 1 " << queryRows << " " << queryCols << " " << queryType << " " << queryDataSize << " " << trainRows << " " << trainCols << " " << trainType << " " << trainDataSize << std::endl;

    std::vector<cv::KeyPoint> queryKeypoints;
    
    std::cout << "cv 2" << std::endl;
    
    cv::Mat queryDescriptors(queryRows, queryCols, queryType);
    if (queryDataSize > 0) {
      memcpy(queryDescriptors.data, queryData, queryDataSize);
    }
    
    cv::Mat trainDescriptors(trainRows, trainCols, trainType);
    if (trainDataSize > 0) {
      memcpy(trainDescriptors.data, trainData, trainDataSize);
    }

    std::cout << "cv 4" << std::endl;

    std::vector<cv::DMatch> matches;
    std::cout << "cv 5 " << queryDescriptors.cols << " " << trainDescriptors.cols << std::endl;
    if (queryDescriptors.cols == trainDescriptors.cols) {
      std::vector< std::vector<cv::DMatch> > knn_matches;
      matcher->knnMatch(queryDescriptors, trainDescriptors, knn_matches, 2);

      for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
          matches.push_back(knn_matches[i][0]);
        }
      }
    }
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
      return a.distance < b.distance;
    });
    
    std::cout << "cv 6" << std::endl;

    if (matches.size() > 0) {
      std::cout << "matches yes " << queryDescriptors.cols << " " << trainDescriptors.cols << " " << matches.size() << std::endl;
    } else {
      std::cout << "matches no " << queryDescriptors.cols << " " << trainDescriptors.cols << " " << matches.size() << std::endl;
    }
    
    {
      *matchIndices = (uint32_t *)malloc(matches.size() * 2 * sizeof(uint32_t));
      for (size_t i = 0; i < matches.size(); i++) {
        (*matchIndices)[i*2] = matches[i].queryIdx;
        (*matchIndices)[i*2+1] = matches[i].trainIdx;
      }
      *matchIndicesSize = matches.size() * 2 * sizeof(uint32_t);
    }
    
    std::cout << "cv 7" << std::endl;
  } catch(cv::Exception& e) {
    std::cout << "exception caught: " << e.what() << std::endl;
  }
  std::cout << "cv 10" << std::endl;
}
EMSCRIPTEN_KEEPALIVE void doGetQr(int width, int height, unsigned char *imgData, float *viewMatrixInverse, float *projectionMatrixInverse, float *qrCodes, unsigned int *numQrCodes) {
    getQr(width, height, imgData, viewMatrixInverse, projectionMatrixInverse, qrCodes, *numQrCodes);
}
}

EMSCRIPTEN_BINDINGS(binding_utils)
{
    register_vector<int>("IntVector");
    register_vector<float>("FloatVector");
    register_vector<double>("DoubleVector");
    register_vector<cv::Point>("PointVector");
    register_vector<cv::Mat>("MatVector");
    register_vector<cv::Rect>("RectVector");
    register_vector<cv::KeyPoint>("KeyPointVector");
    register_vector<cv::DMatch>("DMatchVector");
    register_vector<std::vector<cv::DMatch>>("DMatchVectorVector");


    emscripten::class_<cv::Mat>("Mat")
        .constructor<>()
        .constructor<const Mat&>()
        .constructor<Size, int>()
        .constructor<int, int, int>()
        .constructor<int, int, int, const Scalar&>()
        .constructor(&binding_utils::createMat, allow_raw_pointers())

        .class_function("eye", select_overload<Mat(Size, int)>(&binding_utils::matEye))
        .class_function("eye", select_overload<Mat(int, int, int)>(&binding_utils::matEye))
        .class_function("ones", select_overload<Mat(Size, int)>(&binding_utils::matOnes))
        .class_function("ones", select_overload<Mat(int, int, int)>(&binding_utils::matOnes))
        .class_function("zeros", select_overload<Mat(Size, int)>(&binding_utils::matZeros))
        .class_function("zeros", select_overload<Mat(int, int, int)>(&binding_utils::matZeros))

        .property("rows", &cv::Mat::rows)
        .property("cols", &cv::Mat::cols)
        .property("matSize", &binding_utils::getMatSize)
        .property("step", &binding_utils::getMatStep)
        .property("data", &binding_utils::matData<unsigned char>)
        .property("data8S", &binding_utils::matData<char>)
        .property("data16U", &binding_utils::matData<unsigned short>)
        .property("data16S", &binding_utils::matData<short>)
        .property("data32S", &binding_utils::matData<int>)
        .property("data32F", &binding_utils::matData<float>)
        .property("data64F", &binding_utils::matData<double>)

        .function("elemSize", select_overload<size_t()const>(&cv::Mat::elemSize))
        .function("elemSize1", select_overload<size_t()const>(&cv::Mat::elemSize1))
        .function("channels", select_overload<int()const>(&cv::Mat::channels))
        .function("convertTo", select_overload<void(const Mat&, Mat&, int, double, double)>(&binding_utils::convertTo))
        .function("convertTo", select_overload<void(const Mat&, Mat&, int)>(&binding_utils::convertTo))
        .function("convertTo", select_overload<void(const Mat&, Mat&, int, double)>(&binding_utils::convertTo))
        .function("total", select_overload<size_t()const>(&cv::Mat::total))
        .function("row", select_overload<Mat(int)const>(&cv::Mat::row))
        .function("create", select_overload<void(int, int, int)>(&cv::Mat::create))
        .function("create", select_overload<void(Size, int)>(&cv::Mat::create))
        .function("rowRange", select_overload<Mat(int, int)const>(&cv::Mat::rowRange))
        .function("rowRange", select_overload<Mat(const Range&)const>(&cv::Mat::rowRange))
        .function("copyTo", select_overload<void(const Mat&, Mat&)>(&binding_utils::matCopyTo))
        .function("copyTo", select_overload<void(const Mat&, Mat&, const Mat&)>(&binding_utils::matCopyTo))
        .function("type", select_overload<int()const>(&cv::Mat::type))
        .function("empty", select_overload<bool()const>(&cv::Mat::empty))
        .function("colRange", select_overload<Mat(int, int)const>(&cv::Mat::colRange))
        .function("colRange", select_overload<Mat(const Range&)const>(&cv::Mat::colRange))
        .function("step1", select_overload<size_t(int)const>(&cv::Mat::step1))
        .function("clone", select_overload<Mat()const>(&cv::Mat::clone))
        .function("depth", select_overload<int()const>(&cv::Mat::depth))
        .function("col", select_overload<Mat(int)const>(&cv::Mat::col))
        .function("dot", select_overload<double(const Mat&, const Mat&)>(&binding_utils::matDot))
        .function("mul", select_overload<Mat(const Mat&, const Mat&, double)>(&binding_utils::matMul))
        .function("inv", select_overload<Mat(const Mat&, int)>(&binding_utils::matInv))
        .function("t", select_overload<Mat(const Mat&)>(&binding_utils::matT))
        .function("roi", select_overload<Mat(const Rect&)const>(&cv::Mat::operator()))
        .function("diag", select_overload<Mat(const Mat&, int)>(&binding_utils::matDiag))
        .function("diag", select_overload<Mat(const Mat&)>(&binding_utils::matDiag))
        .function("isContinuous", select_overload<bool()const>(&cv::Mat::isContinuous))
        .function("setTo", select_overload<void(Mat&, const Scalar&)>(&binding_utils::matSetTo))
        .function("setTo", select_overload<void(Mat&, const Scalar&, const Mat&)>(&binding_utils::matSetTo))
        .function("size", select_overload<Size(const Mat&)>(&binding_utils::matSize))

        .function("ptr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<unsigned char>))
        .function("ptr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<unsigned char>))
        .function("ucharPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<unsigned char>))
        .function("ucharPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<unsigned char>))
        .function("charPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<char>))
        .function("charPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<char>))
        .function("shortPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<short>))
        .function("shortPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<short>))
        .function("ushortPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<unsigned short>))
        .function("ushortPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<unsigned short>))
        .function("intPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<int>))
        .function("intPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<int>))
        .function("floatPtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<float>))
        .function("floatPtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<float>))
        .function("doublePtr", select_overload<val(const Mat&, int)>(&binding_utils::matPtr<double>))
        .function("doublePtr", select_overload<val(const Mat&, int, int)>(&binding_utils::matPtr<double>))

        .function("charAt", select_overload<char&(int)>(&cv::Mat::at<char>))
        .function("charAt", select_overload<char&(int, int)>(&cv::Mat::at<char>))
        .function("charAt", select_overload<char&(int, int, int)>(&cv::Mat::at<char>))
        .function("ucharAt", select_overload<unsigned char&(int)>(&cv::Mat::at<unsigned char>))
        .function("ucharAt", select_overload<unsigned char&(int, int)>(&cv::Mat::at<unsigned char>))
        .function("ucharAt", select_overload<unsigned char&(int, int, int)>(&cv::Mat::at<unsigned char>))
        .function("shortAt", select_overload<short&(int)>(&cv::Mat::at<short>))
        .function("shortAt", select_overload<short&(int, int)>(&cv::Mat::at<short>))
        .function("shortAt", select_overload<short&(int, int, int)>(&cv::Mat::at<short>))
        .function("ushortAt", select_overload<unsigned short&(int)>(&cv::Mat::at<unsigned short>))
        .function("ushortAt", select_overload<unsigned short&(int, int)>(&cv::Mat::at<unsigned short>))
        .function("ushortAt", select_overload<unsigned short&(int, int, int)>(&cv::Mat::at<unsigned short>))
        .function("intAt", select_overload<int&(int)>(&cv::Mat::at<int>) )
        .function("intAt", select_overload<int&(int, int)>(&cv::Mat::at<int>) )
        .function("intAt", select_overload<int&(int, int, int)>(&cv::Mat::at<int>) )
        .function("floatAt", select_overload<float&(int)>(&cv::Mat::at<float>))
        .function("floatAt", select_overload<float&(int, int)>(&cv::Mat::at<float>))
        .function("floatAt", select_overload<float&(int, int, int)>(&cv::Mat::at<float>))
        .function("doubleAt", select_overload<double&(int, int, int)>(&cv::Mat::at<double>))
        .function("doubleAt", select_overload<double&(int)>(&cv::Mat::at<double>))
        .function("doubleAt", select_overload<double&(int, int)>(&cv::Mat::at<double>));

    emscripten::value_object<cv::Range>("Range")
        .field("start", &cv::Range::start)
        .field("end", &cv::Range::end);

    emscripten::value_object<cv::TermCriteria>("TermCriteria")
        .field("type", &cv::TermCriteria::type)
        .field("maxCount", &cv::TermCriteria::maxCount)
        .field("epsilon", &cv::TermCriteria::epsilon);

#define EMSCRIPTEN_CV_SIZE(type) \
    emscripten::value_object<type>("#type") \
        .field("width", &type::width) \
        .field("height", &type::height);

    EMSCRIPTEN_CV_SIZE(Size)
    EMSCRIPTEN_CV_SIZE(Size2f)

#define EMSCRIPTEN_CV_POINT(type) \
    emscripten::value_object<type>("#type") \
        .field("x", &type::x) \
        .field("y", &type::y); \

    EMSCRIPTEN_CV_POINT(Point)
    EMSCRIPTEN_CV_POINT(Point2f)

#define EMSCRIPTEN_CV_RECT(type, name) \
    emscripten::value_object<cv::Rect_<type>> (name) \
        .field("x", &cv::Rect_<type>::x) \
        .field("y", &cv::Rect_<type>::y) \
        .field("width", &cv::Rect_<type>::width) \
        .field("height", &cv::Rect_<type>::height);

    EMSCRIPTEN_CV_RECT(int, "Rect")
    EMSCRIPTEN_CV_RECT(float, "Rect2f")

    emscripten::value_object<cv::RotatedRect>("RotatedRect")
        .field("center", &cv::RotatedRect::center)
        .field("size", &cv::RotatedRect::size)
        .field("angle", &cv::RotatedRect::angle);

    function("rotatedRectPoints", select_overload<emscripten::val(const cv::RotatedRect&)>(&binding_utils::rotatedRectPoints));
    function("rotatedRectBoundingRect", select_overload<Rect(const cv::RotatedRect&)>(&binding_utils::rotatedRectBoundingRect));
    function("rotatedRectBoundingRect2f", select_overload<Rect2f(const cv::RotatedRect&)>(&binding_utils::rotatedRectBoundingRect2f));

    emscripten::value_object<cv::KeyPoint>("KeyPoint")
        .field("angle", &cv::KeyPoint::angle)
        .field("class_id", &cv::KeyPoint::class_id)
        .field("octave", &cv::KeyPoint::octave)
        .field("pt", &cv::KeyPoint::pt)
        .field("response", &cv::KeyPoint::response)
        .field("size", &cv::KeyPoint::size);

    emscripten::value_object<cv::DMatch>("DMatch")
        .field("queryIdx", &cv::DMatch::queryIdx)
        .field("trainIdx", &cv::DMatch::trainIdx)
        .field("imgIdx", &cv::DMatch::imgIdx)
        .field("distance", &cv::DMatch::distance);

    emscripten::value_array<cv::Scalar_<double>> ("Scalar")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>())
        .element(emscripten::index<2>())
        .element(emscripten::index<3>());

    emscripten::value_object<binding_utils::MinMaxLoc>("MinMaxLoc")
        .field("minVal", &binding_utils::MinMaxLoc::minVal)
        .field("maxVal", &binding_utils::MinMaxLoc::maxVal)
        .field("minLoc", &binding_utils::MinMaxLoc::minLoc)
        .field("maxLoc", &binding_utils::MinMaxLoc::maxLoc);

    emscripten::value_object<binding_utils::Circle>("Circle")
        .field("center", &binding_utils::Circle::center)
        .field("radius", &binding_utils::Circle::radius);

    emscripten::value_object<cv::Moments >("Moments")
        .field("m00", &cv::Moments::m00)
        .field("m10", &cv::Moments::m10)
        .field("m01", &cv::Moments::m01)
        .field("m20", &cv::Moments::m20)
        .field("m11", &cv::Moments::m11)
        .field("m02", &cv::Moments::m02)
        .field("m30", &cv::Moments::m30)
        .field("m21", &cv::Moments::m21)
        .field("m12", &cv::Moments::m12)
        .field("m03", &cv::Moments::m03)
        .field("mu20", &cv::Moments::mu20)
        .field("mu11", &cv::Moments::mu11)
        .field("mu02", &cv::Moments::mu02)
        .field("mu30", &cv::Moments::mu30)
        .field("mu21", &cv::Moments::mu21)
        .field("mu12", &cv::Moments::mu12)
        .field("mu03", &cv::Moments::mu03)
        .field("nu20", &cv::Moments::nu20)
        .field("nu11", &cv::Moments::nu11)
        .field("nu02", &cv::Moments::nu02)
        .field("nu30", &cv::Moments::nu30)
        .field("nu21", &cv::Moments::nu21)
        .field("nu12", &cv::Moments::nu12)
        .field("nu03", &cv::Moments::nu03);

    emscripten::value_object<cv::Exception>("Exception")
        .field("code", &cv::Exception::code)
        .field("msg", &binding_utils::getExceptionMsg, &binding_utils::setExceptionMsg);

    function("exceptionFromPtr", &binding_utils::exceptionFromPtr, allow_raw_pointers());

#ifdef HAVE_OPENCV_IMGPROC
    function("minEnclosingCircle", select_overload<binding_utils::Circle(const cv::Mat&)>(&binding_utils::minEnclosingCircle));

    function("floodFill", select_overload<int(cv::Mat&, cv::Mat&, Point, Scalar, emscripten::val, Scalar, Scalar, int)>(&binding_utils::floodFill_wrapper));

    function("floodFill", select_overload<int(cv::Mat&, cv::Mat&, Point, Scalar, emscripten::val, Scalar, Scalar)>(&binding_utils::floodFill_wrapper_1));

    function("floodFill", select_overload<int(cv::Mat&, cv::Mat&, Point, Scalar, emscripten::val, Scalar)>(&binding_utils::floodFill_wrapper_2));

    function("floodFill", select_overload<int(cv::Mat&, cv::Mat&, Point, Scalar, emscripten::val)>(&binding_utils::floodFill_wrapper_3));

    function("floodFill", select_overload<int(cv::Mat&, cv::Mat&, Point, Scalar)>(&binding_utils::floodFill_wrapper_4));
#endif

    function("minMaxLoc", select_overload<binding_utils::MinMaxLoc(const cv::Mat&, const cv::Mat&)>(&binding_utils::minMaxLoc));

    function("minMaxLoc", select_overload<binding_utils::MinMaxLoc(const cv::Mat&)>(&binding_utils::minMaxLoc_1));

#ifdef HAVE_OPENCV_IMGPROC
    function("morphologyDefaultBorderValue", &cv::morphologyDefaultBorderValue);
#endif

    function("CV_MAT_DEPTH", &binding_utils::cvMatDepth);

#ifdef HAVE_OPENCV_VIDEO
    function("CamShift", select_overload<emscripten::val(const cv::Mat&, Rect&, TermCriteria)>(&binding_utils::CamShiftWrapper));

    function("meanShift", select_overload<emscripten::val(const cv::Mat&, Rect&, TermCriteria)>(&binding_utils::meanShiftWrapper));
#endif

    function("getBuildInformation", &binding_utils::getBuildInformation);

#ifdef HAVE_PTHREADS_PF
    function("parallel_pthreads_set_threads_num", &cv::parallel_pthreads_set_threads_num);
    function("parallel_pthreads_get_threads_num", &cv::parallel_pthreads_get_threads_num);
#endif

#ifdef TEST_WASM_INTRIN
    function("test_hal_intrin_uint8", &binding_utils::test_hal_intrin_uint8);
    function("test_hal_intrin_int8", &binding_utils::test_hal_intrin_int8);
    function("test_hal_intrin_uint16", &binding_utils::test_hal_intrin_uint16);
    function("test_hal_intrin_int16", &binding_utils::test_hal_intrin_int16);
    function("test_hal_intrin_uint32", &binding_utils::test_hal_intrin_uint32);
    function("test_hal_intrin_int32", &binding_utils::test_hal_intrin_int32);
    function("test_hal_intrin_uint64", &binding_utils::test_hal_intrin_uint64);
    function("test_hal_intrin_int64", &binding_utils::test_hal_intrin_int64);
    function("test_hal_intrin_float32", &binding_utils::test_hal_intrin_float32);
    function("test_hal_intrin_float64", &binding_utils::test_hal_intrin_float64);
    function("test_hal_intrin_all", &binding_utils::test_hal_intrin_all);
#endif

    constant("CV_8UC1", CV_8UC1);
    constant("CV_8UC2", CV_8UC2);
    constant("CV_8UC3", CV_8UC3);
    constant("CV_8UC4", CV_8UC4);

    constant("CV_8SC1", CV_8SC1);
    constant("CV_8SC2", CV_8SC2);
    constant("CV_8SC3", CV_8SC3);
    constant("CV_8SC4", CV_8SC4);

    constant("CV_16UC1", CV_16UC1);
    constant("CV_16UC2", CV_16UC2);
    constant("CV_16UC3", CV_16UC3);
    constant("CV_16UC4", CV_16UC4);

    constant("CV_16SC1", CV_16SC1);
    constant("CV_16SC2", CV_16SC2);
    constant("CV_16SC3", CV_16SC3);
    constant("CV_16SC4", CV_16SC4);

    constant("CV_32SC1", CV_32SC1);
    constant("CV_32SC2", CV_32SC2);
    constant("CV_32SC3", CV_32SC3);
    constant("CV_32SC4", CV_32SC4);

    constant("CV_32FC1", CV_32FC1);
    constant("CV_32FC2", CV_32FC2);
    constant("CV_32FC3", CV_32FC3);
    constant("CV_32FC4", CV_32FC4);

    constant("CV_64FC1", CV_64FC1);
    constant("CV_64FC2", CV_64FC2);
    constant("CV_64FC3", CV_64FC3);
    constant("CV_64FC4", CV_64FC4);

    constant("CV_8U", CV_8U);
    constant("CV_8S", CV_8S);
    constant("CV_16U", CV_16U);
    constant("CV_16S", CV_16S);
    constant("CV_32S",  CV_32S);
    constant("CV_32F", CV_32F);
    constant("CV_64F", CV_64F);

    constant("INT_MIN", INT_MIN);
    constant("INT_MAX", INT_MAX);
}

// hack
#include "../../../opencv_contrib/modules/xfeatures2d/src/surf.cpp"
