#include "bundle.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace at {
namespace {

void require(bool ok, const std::string& msg) {
    if (!ok) throw std::runtime_error(msg);
}

int asInt(const rapidjson::Value& o, const char* key) {
    require(o.HasMember(key) && o[key].IsNumber(), std::string("manifest missing number: ") + key);
    return o[key].GetInt();
}
double asDouble(const rapidjson::Value& o, const char* key) {
    require(o.HasMember(key) && o[key].IsNumber(), std::string("manifest missing number: ") + key);
    return o[key].GetDouble();
}
std::string asString(const rapidjson::Value& o, const char* key) {
    require(o.HasMember(key) && o[key].IsString(), std::string("manifest missing string: ") + key);
    return o[key].GetString();
}

size_t elementCount(const std::vector<int>& shape) {
    size_t n = 1;
    for (int d : shape) {
        require(d > 0, "invalid tensor shape");
        if (n > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
            throw std::runtime_error("tensor element-count overflow");
        }
        n *= static_cast<size_t>(d);
    }
    return n;
}

}  // namespace

std::string readTextFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

std::vector<uint8_t> readBinaryFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    auto len = f.tellg();
    require(len >= 0, "tellg failed for " + path);
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> out(static_cast<size_t>(len));
    if (!out.empty()) f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    require(static_cast<bool>(f) || f.eof(), "read failed for " + path);
    return out;
}

void atomicWrite(const std::string& path, const void* data, size_t size) {
    const std::string tmp = path + ".tmp";
    {
        FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) throw std::runtime_error("fopen failed: " + tmp + ": " + std::strerror(errno));
        size_t wrote = std::fwrite(data, 1, size, f);
        if (wrote != size) {
            std::fclose(f);
            std::remove(tmp.c_str());
            throw std::runtime_error("short write: " + tmp);
        }
        if (std::fflush(f) != 0 || ::fsync(::fileno(f)) != 0) {
            std::fclose(f);
            std::remove(tmp.c_str());
            throw std::runtime_error("fsync failed: " + tmp);
        }
        if (std::fclose(f) != 0) {
            std::remove(tmp.c_str());
            throw std::runtime_error("fclose failed: " + tmp);
        }
    }
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("atomic rename failed: " + path);
    }
}

std::string Bundle::path(const std::string& rel) const {
    if (rel.empty() || rel[0] == '/' || rel.find("..") != std::string::npos) {
        throw std::runtime_error("unsafe bundle relative path");
    }
    return root + "/" + rel;
}

const TensorData& Bundle::tensor(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("missing tensor slot " + name);
    return it->second;
}

Bundle Bundle::load(const std::string& rootDir) {
    Bundle b;
    b.root = rootDir;
    const std::string json = readTextFile(rootDir + "/manifest.json");
    b.manifest.Parse(json.c_str(), json.size());
    require(!b.manifest.HasParseError() && b.manifest.IsObject(), "invalid manifest.json");

    b.schema = asString(b.manifest, "schema");
    require(b.schema == "android_trainer_bundle_v2", "unsupported bundle schema " + b.schema);
    b.checkpointSha256 = asString(b.manifest, "checkpoint_sha256");
    b.trainBinSha256 = asString(b.manifest, "train_bin_sha256");
    b.modelStateSha256 = asString(b.manifest, "model_state_sha256");
    b.ropeStyle = asString(b.manifest, "rope_style");
    require(b.ropeStyle == "half_split" || b.ropeStyle == "interleaved" || b.ropeStyle == "auto", "unsupported RoPE style");
    require(b.manifest.HasMember("parameter_count") && b.manifest["parameter_count"].IsInt64(), "bad parameter_count");
    b.parameterCount = b.manifest["parameter_count"].GetInt64();
    require(b.parameterCount == 19145088, "parameter-count drift");

    require(b.manifest.HasMember("config") && b.manifest["config"].IsObject(), "missing config");
    const auto& c = b.manifest["config"];
    b.config.vocabSize = asInt(c, "vocab_size");
    b.config.seqLen = asInt(c, "seq_len");
    b.config.dModel = asInt(c, "d_model");
    b.config.nLayers = asInt(c, "n_layers");
    b.config.nHeads = asInt(c, "n_heads");
    b.config.nKvHeads = asInt(c, "n_kv_heads");
    b.config.headDim = asInt(c, "head_dim");
    b.config.dFf = asInt(c, "d_ff");
    b.config.ropeTheta = asDouble(c, "rope_theta");
    b.config.rmsNormEps = asDouble(c, "rms_norm_eps");
    require(b.config.vocabSize == 14000 && b.config.seqLen == 256 && b.config.dModel == 384 &&
            b.config.nLayers == 8 && b.config.nHeads == 6 && b.config.nKvHeads == 2 &&
            b.config.headDim == 64 && b.config.dFf == 1152,
            "frozen Model #0001 geometry mismatch");
    require(std::isfinite(b.config.ropeTheta) && b.config.ropeTheta > 1.0, "invalid rope_theta");
    require(std::isfinite(b.config.rmsNormEps) && b.config.rmsNormEps > 0.0, "invalid rms_norm_eps");

    require(b.manifest.HasMember("optimizer") && b.manifest["optimizer"].IsObject(), "missing optimizer");
    const auto& a = b.manifest["optimizer"];
    b.adam.beta1 = asDouble(a, "beta1");
    b.adam.beta2 = asDouble(a, "beta2");
    b.adam.eps = asDouble(a, "eps");
    b.adam.gateLr = asDouble(a, "gate_lr");
    require(a.HasMember("slot_weight_decay") && a["slot_weight_decay"].IsObject(),
            "optimizer missing slot_weight_decay");
    for (auto it = a["slot_weight_decay"].MemberBegin(); it != a["slot_weight_decay"].MemberEnd(); ++it) {
        require(it->name.IsString() && it->value.IsNumber(), "bad slot_weight_decay entry");
        const double wd = it->value.GetDouble();
        require(std::isfinite(wd) && wd >= 0.0, "invalid slot weight decay");
        b.adam.slotWeightDecay[it->name.GetString()] = wd;
    }
    require(b.adam.beta1 > 0 && b.adam.beta1 < 1 && b.adam.beta2 > 0 && b.adam.beta2 < 1 &&
            b.adam.eps > 0 && b.adam.gateLr > 0,
            "invalid AdamW config");

    require(b.manifest.HasMember("reference") && b.manifest["reference"].IsObject(), "missing reference");
    const auto& r = b.manifest["reference"];
    b.reference.loss = asDouble(r, "loss");
    b.reference.globalGradNorm = asDouble(r, "global_grad_norm");
    b.reference.clipCoef = asDouble(r, "clip_coef");
    require(std::isfinite(b.reference.loss) && std::isfinite(b.reference.globalGradNorm), "nonfinite reference");

    require(b.manifest.HasMember("sample") && b.manifest["sample"].IsObject(), "missing sample");
    const auto& s = b.manifest["sample"];
    require(asInt(s, "token_count") == 257, "gate sample must contain 257 tokens");
    auto sb = readBinaryFile(b.path(asString(s, "tokens_file")));
    require(sb.size() == 257u * sizeof(int32_t), "sample byte-size mismatch");
    b.sampleTokens.resize(257);
    std::memcpy(b.sampleTokens.data(), sb.data(), sb.size());
    for (int32_t id : b.sampleTokens) require(id >= 0 && id < 14000, "sample token out of vocab");

    require(b.manifest.HasMember("tensors") && b.manifest["tensors"].IsObject(), "missing tensors");
    const auto& ts = b.manifest["tensors"];
    for (auto it = ts.MemberBegin(); it != ts.MemberEnd(); ++it) {
        require(it->name.IsString() && it->value.IsObject(), "bad tensor manifest entry");
        const std::string name = it->name.GetString();
        const auto& e = it->value;
        require(asString(e, "dtype") == "f32", "only f32 gate tensors supported");
        require(e.HasMember("shape") && e["shape"].IsArray(), "tensor missing shape: " + name);
        TensorData td;
        for (const auto& d : e["shape"].GetArray()) {
            require(d.IsInt() && d.GetInt() > 0, "bad tensor dimension: " + name);
            td.shape.push_back(d.GetInt());
        }
        td.sha256 = asString(e, "sha256");
        const int64_t nbytes = e["nbytes"].GetInt64();
        require(nbytes > 0 && static_cast<uint64_t>(nbytes) == elementCount(td.shape) * sizeof(float),
                "tensor nbytes mismatch: " + name);
        auto bytes = readBinaryFile(b.path(asString(e, "path")));
        require(static_cast<int64_t>(bytes.size()) == nbytes, "tensor file length mismatch: " + name);
        td.data.resize(bytes.size() / sizeof(float));
        std::memcpy(td.data.data(), bytes.data(), bytes.size());
        for (float x : td.data) require(std::isfinite(x), "nonfinite model tensor: " + name);
        b.tensors.emplace(name, std::move(td));
    }

    require(b.tensors.size() == 1u + 8u * 9u + 1u, "unexpected normalized tensor-slot count");
    require(b.adam.slotWeightDecay.size() == b.tensors.size(),
            "slot_weight_decay count does not match tensor slots");
    for (const auto& kv : b.tensors) {
        require(b.adam.slotWeightDecay.find(kv.first) != b.adam.slotWeightDecay.end(),
                "missing slot weight decay: " + kv.first);
    }
    return b;
}

std::string jsonEscape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '\\': o += "\\\\"; break;
            case '"': o += "\\\""; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o += buf;
                } else o += static_cast<char>(c);
        }
    }
    return o;
}

}  // namespace at
