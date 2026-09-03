#include <jni.h>
#include <string>
#include "model0001_gate.hpp"
#include "native_opencl_conformance.hpp"
#include "bundle.hpp"
#include "native_opencl_trainer.hpp"

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


extern "C" JNIEXPORT jstring JNICALL
Java_dev_riszn_androidtrainer_MainActivity_nativeRunLrPilot(
        JNIEnv* env, jclass, jstring bundleDir, jstring pilotDir, jstring workDir) {
    try {
        at::Bundle bundle = at::Bundle::load(jstr(env, bundleDir));
        at::NativePilotResult result = at::runNativeModel0001LrPilot(
            bundle, jstr(env, pilotDir), jstr(env, workDir));
        return ret(env, result.json);
    } catch (const std::exception& error) {
        return ret(env,
            std::string("{\"status\":\"FAIL_BRIDGE_EXCEPTION\",\"schema\":") +
            "\"model0001_v3_lr_pilot_report_v1\",\"error\":\"" +
            at::jsonEscape(error.what()) +
            "\",\"production_lr_locked\":false,\"test_split_used\":false,\"pass\":false}");
    }
}


extern "C" JNIEXPORT jstring JNICALL
Java_dev_riszn_androidtrainer_MainActivity_nativeRunStage(
        JNIEnv* env, jclass, jstring bundleDir, jstring stageDir, jstring workDir) {
    try {
        at::Bundle bundle = at::Bundle::load(jstr(env, bundleDir));
        at::NativeStageResult result = at::runNativeModel0001Stage(
            bundle, jstr(env, stageDir), jstr(env, workDir));
        return ret(env, result.json);
    } catch (const std::exception& error) {
        return ret(env,
            std::string("{\"status\":\"FAIL_BRIDGE_EXCEPTION\",\"schema\":") +
            "\"model0001_native_stage_report_v1\",\"error\":\"" +
            at::jsonEscape(error.what()) +
            "\",\"production_lr_locked\":true,\"test_split_used\":false,\"pass\":false}");
    }
}


extern "C" JNIEXPORT jstring JNICALL
Java_dev_riszn_androidtrainer_MainActivity_nativeRunF2SftLrPilot(
        JNIEnv* env, jclass, jstring bundleDir, jstring pilotDir, jstring workDir) {
    try {
        at::Bundle bundle = at::Bundle::load(jstr(env, bundleDir));
        at::NativePilotResult result = at::runNativeModel0001SftLrPilot(
            bundle, jstr(env, pilotDir), jstr(env, workDir));
        return ret(env, result.json);
    } catch (const std::exception& error) {
        return ret(env,
            std::string("{\"status\":\"FAIL_BRIDGE_EXCEPTION\",\"schema\":") +
            "\"model0001_f2_sft_lr_pilot_report_v1\",\"error\":\"" +
            at::jsonEscape(error.what()) +
            "\",\"production_lr_locked\":false,\"test_split_used\":false,\"pass\":false}");
    }
}


extern "C" JNIEXPORT jstring JNICALL
Java_dev_riszn_androidtrainer_MainActivity_nativeRunF2SftStage(
        JNIEnv* env, jclass, jstring bundleDir, jstring stageDir, jstring workDir) {
    try {
        at::Bundle bundle = at::Bundle::load(jstr(env, bundleDir));
        at::NativeStageResult result = at::runNativeModel0001SftStage(
            bundle, jstr(env, stageDir), jstr(env, workDir));
        return ret(env, result.json);
    } catch (const std::exception& error) {
        return ret(env,
            std::string("{\"status\":\"FAIL_BRIDGE_EXCEPTION\",\"schema\":") +
            "\"model0001_f2_sft_stage_report_v1\",\"error\":\"" +
            at::jsonEscape(error.what()) +
            "\",\"production_lr_locked\":true,\"test_split_used\":false,\"pass\":false}");
    }
}
