#pragma once
#include <string>
namespace at {
std::string probeBackendsJson();
std::string validateBundleJson(const std::string& bundleDir);
std::string runModel0001GateJson(const std::string& bundleDir, const std::string& workDir, float thermalHeadroom);
}
