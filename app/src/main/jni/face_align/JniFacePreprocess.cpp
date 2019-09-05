//
// Created by LW on 2019/9/3.
//

#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>
#include "FacePreprocess.h"

// 将bitmap转为mat
void bitmap2mat(JNIEnv *env, jobject &bitmap, cv::Mat &dst,
                            bool needUnPremultiplyAlpha) {
    AndroidBitmapInfo info;
    void *pixels = 0;

    try {
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                  info.format == ANDROID_BITMAP_FORMAT_RGB_565);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);

        dst.create(info.height, info.width, CV_8UC4);

        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            //   LOG(INFO) << "nBitmapToMat: RGBA_8888 -> CV_8UC4";
            cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);

            if (needUnPremultiplyAlpha)
                cvtColor(tmp, dst, cv::COLOR_mRGBA2RGBA);
            else
                tmp.copyTo(dst);

        } else {
            //   LOG(INFO) << "nBitmapToMat: RGB_565 -> CV_8UC4";
            cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
            cvtColor(tmp, dst, cv::COLOR_BGR5652RGBA);
        }

        AndroidBitmap_unlockPixels(env, bitmap);

        return;
    } catch (const cv::Exception &e) {
        AndroidBitmap_unlockPixels(env, bitmap);

        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je)
            je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);

        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nBitmapToMat}");
        return;
    }
}


//将Mat转换为bitmap
jobject mat2bitmap(JNIEnv *env, cv::Mat &src, bool needPremultiplyAlpha, jobject bitmap_config) {

    jclass java_bitmap_class = (jclass) env->FindClass("android/graphics/Bitmap");
    jmethodID mid = env->GetStaticMethodID(java_bitmap_class, "createBitmap",
                                           "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jobject bitmap = env->CallStaticObjectMethod(java_bitmap_class,
                                                 mid, src.size().width, src.size().height,
                                                 bitmap_config);
    AndroidBitmapInfo info;
    void *pixels = 0;

    try {
        CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
        CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
        CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
        CV_Assert(pixels);

        if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
            cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
            if (src.type() == CV_8UC1) {
                cvtColor(src, tmp, cv::COLOR_GRAY2RGBA);
            } else if (src.type() == CV_8UC3) {
                cvtColor(src, tmp, cv::COLOR_RGB2BGRA);
            } else if (src.type() == CV_8UC4) {
                if (needPremultiplyAlpha) {
                    cvtColor(src, tmp, cv::COLOR_RGBA2mRGBA);
                } else {
                    src.copyTo(tmp);
                }
            }
        } else {
            // info.format == ANDROID_BITMAP_FORMAT_RGB_565
            cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
            if (src.type() == CV_8UC1) {
                cvtColor(src, tmp, cv::COLOR_GRAY2BGR565);
            } else if (src.type() == CV_8UC3) {
                cvtColor(src, tmp, cv::COLOR_RGB2BGR565);
            } else if (src.type() == CV_8UC4) {
                cvtColor(src, tmp, cv::COLOR_RGBA2BGR565);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return bitmap;
    } catch (cv::Exception e) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("org/opencv/core/CvException");
        if (!je) je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, e.what());
        return nullptr;
    } catch (...) {
        AndroidBitmap_unlockPixels(env, bitmap);
        jclass je = env->FindClass("java/lang/Exception");
        env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
        return nullptr;
    }
}



extern "C" JNIEXPORT jobject JNICALL
Java_pp_facerecognizer_align_FacePreprocess_facePreprocess(JNIEnv *env, jobject instance, \
        jobject _bitmap, jobjectArray _jlandmark) {


    jint i,j;

    int size = env->GetArrayLength( _jlandmark);//获得行数

    jarray myarray = (jarray)env->GetObjectArrayElement(_jlandmark, 0);

    int col =env->GetArrayLength(myarray); //获得列数

    jfloat _landmark[size][col];

    for (i = 0; i < size; i++){

        myarray = (jarray)env->GetObjectArrayElement(_jlandmark, i);

        jfloat *coldata = env->GetFloatArrayElements((jfloatArray)myarray, 0 );

        for (j=0; j<col; j++) {
            _landmark [i] [j] = coldata[j];
        }

    }


    cv::Mat image(112, 112, CV_32FC1);
    bitmap2mat(env, _bitmap, image, false);


    float _src[5][2]={
            { 30.2946f + 8.0f, 51.6963f},
            { 65.5318f + 8.0f, 51.5014f},
            { 48.0252f + 8.0f, 71.7366f},
            { 33.5493f + 8.0f, 92.3655f},
            { 62.7299f + 8.0f, 92.2041f}
    };

    cv::Mat src(5, 2, CV_32FC1, _src);
    memcpy(src.data, _src, 2 * 5 * sizeof(float));

    cv::Mat dst(5, 2, CV_32FC1, _landmark);
    memcpy(dst.data, _landmark, 2 * 5 * sizeof(float));

    cv::Mat M = FacePreprocess::similarTransform(dst, src);  // skimage.transform.SimilarityTransform

    cv::Mat warpImg;

    cv::warpPerspective(image, warpImg, M, cv::Size(112, 112));

    jclass java_bitmap_class = (jclass) env->FindClass("android/graphics/Bitmap");
    jmethodID mid = env->GetMethodID(java_bitmap_class, "getConfig",
                                     "()Landroid/graphics/Bitmap$Config;");
    jobject bitmap_config = env->CallObjectMethod(_bitmap, mid);

    jobject bitmap = mat2bitmap(env, warpImg, false, bitmap_config);

    return bitmap;

}



















