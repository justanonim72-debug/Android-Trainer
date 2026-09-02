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

std::string probeNativeOpenClJson();

}  // namespace at
