#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <rapidjson/document.h>

namespace at {

struct TensorData {
    std::vector<int> shape;
    std::vector<float> data;
    std::string sha256;
};

struct ModelConfig {
    int vocabSize = 0;
    int seqLen = 0;
    int dModel = 0;
    int nLayers = 0;
    int nHeads = 0;
    int nKvHeads = 0;
    int headDim = 0;
    int dFf = 0;
    double ropeTheta = 0.0;
    double rmsNormEps = 0.0;
};

struct AdamConfig {
    double beta1 = 0.0;
    double beta2 = 0.0;
    double eps = 0.0;
    double gateLr = 0.0;
    std::map<std::string, double> slotWeightDecay;
};

struct ReferenceProbe {
    double loss = 0.0;
    double globalGradNorm = 0.0;
    double clipCoef = 0.0;
};

class Bundle {
public:
    static Bundle load(const std::string& root);

    const TensorData& tensor(const std::string& name) const;
    std::string path(const std::string& rel) const;

    std::string root;
    std::string schema;
    std::string checkpointSha256;
    std::string trainBinSha256;
    std::string modelStateSha256;
    std::string ropeStyle;
    int64_t parameterCount = 0;
    ModelConfig config;
    AdamConfig adam;
    ReferenceProbe reference;
    std::vector<int32_t> sampleTokens;
    std::map<std::string, TensorData> tensors;

    // Keep exact manifest for detailed parity probes.
    rapidjson::Document manifest;
};

std::string readTextFile(const std::string& path);
std::vector<uint8_t> readBinaryFile(const std::string& path);
void atomicWrite(const std::string& path, const void* data, size_t size);
std::string jsonEscape(const std::string& s);

}  // namespace at
