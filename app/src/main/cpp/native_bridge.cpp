#include <jni.h>
#include <string>
#include "model0001_gate.hpp"
#include "native_opencl_conformance.hpp"

namespace {
std::string jstr(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* p = env->GetStringUTFChars(s, nullptr);
    std::string out = p ? p : "";
    if (p) env->ReleaseStringUTFChars(s, p);
    return out;
}
jstring ret(JNIEnv* env, const std::string& s) {
    return env->NewStringUTF(s.c_str());
}
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_riszn_androidtrainer_MainActivity_nativeProbe(JNIEnv* env, jclass) {
    return ret(env, at::probeBackendsJson());
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_riszn_androidtrainer_MainActivity_nativeOpenClConformance(JNIEnv* env, jclass) {
    return ret(env, at::runNativeOpenClConformanceJson());
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_riszn_androidtrainer_MainActivity_nativeValidateBundle(
        JNIEnv* env, jclass, jstring bundleDir) {
    return ret(env, at::validateBundleJson(jstr(env, bundleDir)));
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_riszn_androidtrainer_MainActivity_nativeRunGate(
        JNIEnv* env, jclass, jstring bundleDir, jstring workDir, jfloat thermalHeadroom) {
    return ret(env, at::runModel0001GateJson(
        jstr(env, bundleDir), jstr(env, workDir), static_cast<float>(thermalHeadroom)));
}
