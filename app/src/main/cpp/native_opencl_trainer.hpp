#pragma once

#include "bundle.hpp"

#include <functional>
#include <string>

namespace at {

struct NativeGateResult {
    bool pass = false;
    std::string json;
};

// The callback is invoked only after native gates 1-5 have passed.  This keeps
// the sustained CPU/GPU benchmark behind the correctness and checkpoint gates.
NativeGateResult runNativeModel0001Gate(
    const Bundle& bundle,
    const std::string& workDir,
    const std::function<double()>& cpuBaselineTokensPerSecond);


struct NativePilotResult {
    bool pass = false;
    std::string json;
};

// Runs the locked Foundation-v3 LR pilot from the immutable CPT-v2 bundle.
// The pilot package contains data + protocol only; each LR candidate resets
// source weights and Adam moments before any update.
NativePilotResult runNativeModel0001LrPilot(
    const Bundle& bundle,
    const std::string& pilotRoot,
    const std::string& workDir);

NativePilotResult runNativeModel0001SftLrPilot(
    const Bundle& bundle,
    const std::string& pilotRoot,
    const std::string& workDir);

struct NativeStageResult {
    bool pass = false;
    std::string json;
};

// Runs or resumes the recipe-driven Foundation-v3 production stage.
// The stage package is data+recipe only; source weights come from the same
// immutable CPT-v2 .atb bundle used by the accepted GPU gate.
NativeStageResult runNativeModel0001Stage(
    const Bundle& bundle,
    const std::string& stageRoot,
    const std::string& workDir);

std::string probeNativeOpenClJson();

}  // namespace at
