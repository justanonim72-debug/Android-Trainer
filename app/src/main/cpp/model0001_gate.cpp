#include "model0001_gate.hpp"
#include "bundle.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

#include "OpGrad.hpp"
#include "ParameterOptimizer.hpp"
#include "core/Backend.hpp"
#include "core/TensorUtils.hpp"

#include <CL/cl.h>
#include <dlfcn.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace MNN;
using namespace MNN::Express;

namespace at {
namespace {

thread_local std::string gStagePath;

void markStage(const std::string& stage) {
    if (gStagePath.empty()) return;
    try {
        atomicWrite(gStagePath, stage.data(), stage.size());
    } catch (...) {
        // Crash diagnostics must never alter gate behavior.
    }
}

constexpr int S = 256;
constexpr int V = 14000;
constexpr int D = 384;
constexpr int HQ = 6;
constexpr int HKV = 2;
constexpr int HD = 64;
constexpr int FF = 1152;

void req(bool v, const std::string& m) { if (!v) throw std::runtime_error(m); }

std::string typeName(MNNForwardType t) {
    switch (t) {
        case MNN_FORWARD_CPU: return "CPU";
        case MNN_FORWARD_OPENCL: return "OPENCL";
        case MNN_FORWARD_VULKAN: return "VULKAN";
        default: return "TYPE_" + std::to_string(static_cast<int>(t));
    }
}

VARP scalar(float x) { return _Scalar<float>(x); }

VARP makeParam(const TensorData& t, const std::string& name) {
    auto p = _TrainableParam(t.data.data(), t.shape, NHWC, halide_type_of<float>());
    p->setName(name);
    return p;
}

VARP rmsNorm(VARP x, VARP w, float eps) {
    auto ms = _ReduceMean(_Square(x), {-1}, true);
    return x * _Rsqrt(ms + scalar(eps)) * w;
}

VARP linear(VARP x, VARP w) {
    return _MatMul(x, w, false, true);
}

VARP rotateHalf(VARP x, const std::string& style) {
    if (style == "half_split") {
        auto halves = _Split(x, {HD / 2, HD / 2}, 3);
        req(halves.size() == 2, "RoPE split failed");
        return _Concat({_Negative(halves[1]), halves[0]}, 3);
    }
    // Interleaved [-x1,x0,-x3,x2,...] without GatherV2.
    //
    // MNN 3.6.1 GatherGrad reconstructs axis-0 gradients through ScatterNd
    // with an extra Unsqueeze, and the train tree has no gradient regression
    // test covering this path.  Forward was numerically exact while the CPU
    // backward showed near-2.0 relative probe errors.  Reshape each even/odd
    // pair, split the last axis, then concatenate {-odd, even}; this is exactly
    // the same RoPE permutation/sign transform and uses only Reshape/Slice/
    // Concat/Neg gradients that are explicitly registered in the pinned MNN.
    // K has HKV heads before repeatKv, Q has HQ heads. Preserve the runtime
    // head count instead of hard-coding it in the reshape.
    auto info = x->getInfo();
    req(info && info->dim.size() == 4 && info->dim[3] == HD, "RoPE input shape invalid");
    const int heads = info->dim[1];
    auto pairs = _Reshape(x, {1, heads, S, HD / 2, 2});
    auto eo = _Split(pairs, {1, 1}, 4);
    req(eo.size() == 2, "RoPE interleaved pair split failed");
    return _Reshape(_Concat({_Negative(eo[1]), eo[0]}, 4), {1, heads, S, HD});
}

std::pair<VARP,VARP> rope(VARP q, VARP k, const Bundle& b) {
    std::vector<float> cosv(S * HD), sinv(S * HD);
    for (int pos = 0; pos < S; ++pos) {
        for (int i = 0; i < HD / 2; ++i) {
            const double inv = std::pow(b.config.ropeTheta, -2.0 * i / HD);
            const float c = static_cast<float>(std::cos(pos * inv));
            const float s = static_cast<float>(std::sin(pos * inv));
            if (b.ropeStyle == "half_split") {
                cosv[pos * HD + i] = c; cosv[pos * HD + i + HD/2] = c;
                sinv[pos * HD + i] = s; sinv[pos * HD + i + HD/2] = s;
            } else {
                cosv[pos * HD + 2*i] = c; cosv[pos * HD + 2*i + 1] = c;
                sinv[pos * HD + 2*i] = s; sinv[pos * HD + 2*i + 1] = s;
            }
        }
    }
    auto c = _Const(cosv.data(), {1,1,S,HD}, NHWC);
    auto s = _Const(sinv.data(), {1,1,S,HD}, NHWC);
    return {q * c + rotateHalf(q,b.ropeStyle) * s,
            k * c + rotateHalf(k,b.ropeStyle) * s};
}

VARP repeatKv(VARP x) {
    // Preserve exact GQA repeat semantics while staying inside MNN 3.6.1's
    // registered autograd surface. OpType_Tile has no gradient registration in
    // the pinned MNN train module, while BroadcastTo does. Broadcasting
    // [1,2,1,S,64] -> [1,2,3,S,64] and then reshaping is elementwise identical
    // to Tile(reps=[1,1,3,1,1]); its backward is the required sum over repeats.
    auto e = _Reshape(x, {1,HKV,1,S,HD});
    int32_t shapeData[5] = {1,HKV,HQ/HKV,S,HD};
    auto shape = _Const(shapeData, {5}, NHWC, halide_type_of<int32_t>());
    return _Reshape(_BroadcastTo(e, shape), {1,HQ,S,HD});
}

VARP causalMask() {
    std::vector<float> m(S*S);
    for (int i=0;i<S;++i) for(int j=0;j<S;++j) m[i*S+j]=(j<=i)?0.0f:-1.0e9f;
    return _Const(m.data(), {1,1,S,S}, NHWC);
}

struct Graph {
    VARP tokenInput;
    VARP targetInput;
    VARP embeddingLookup;
    VARP logits;
    VARP loss;
    std::map<std::string,VARP> params;
};

Graph buildGraph(const Bundle& b) {
    Graph g;
    g.tokenInput = _Input({S}, NHWC, halide_type_of<int32_t>());
    g.targetInput = _Input({S}, NHWC, halide_type_of<int32_t>());
    g.tokenInput->setName("tokens");
    g.targetInput->setName("targets");

    auto addP=[&](const std::string& n) {
        auto p=makeParam(b.tensor(n),n);
        g.params.emplace(n,p);
        return p;
    };

    auto embed=addP("tok_embeddings.weight");
    auto axis0=_Scalar<int32_t>(0);
    auto x=_GatherV2(embed,g.tokenInput,axis0); // [S,D]
    g.embeddingLookup=x;
    g.embeddingLookup->setName("embedding_lookup");
    auto mask=causalMask();

    for(int l=0;l<8;++l) {
        const std::string pre="layers."+std::to_string(l)+".";
        auto an=addP(pre+"attn_norm.weight");
        auto qw=addP(pre+"q_proj.weight");
        auto kw=addP(pre+"k_proj.weight");
        auto vw=addP(pre+"v_proj.weight");
        auto ow=addP(pre+"o_proj.weight");
        auto fn=addP(pre+"ffn_norm.weight");
        auto gw=addP(pre+"gate_proj.weight");
        auto uw=addP(pre+"up_proj.weight");
        auto dw=addP(pre+"down_proj.weight");

        auto h=rmsNorm(x,an,static_cast<float>(b.config.rmsNormEps));
        auto q=_Transpose(_Reshape(linear(h,qw),{1,S,HQ,HD}),{0,2,1,3});
        auto k=_Transpose(_Reshape(linear(h,kw),{1,S,HKV,HD}),{0,2,1,3});
        auto v=_Transpose(_Reshape(linear(h,vw),{1,S,HKV,HD}),{0,2,1,3});
        auto rk=rope(q,k,b); q=rk.first; k=rk.second;
        k=repeatKv(k); v=repeatKv(v);
        auto scores=_BatchMatMul(q,k,false,true)*scalar(1.0f/std::sqrt(static_cast<float>(HD))) + mask;
        auto prob=_Softmax(scores,-1);
        auto a=_BatchMatMul(prob,v,false,false);
        a=_Reshape(_Transpose(a,{0,2,1,3}),{S,D});
        x=x+linear(a,ow);

        h=rmsNorm(x,fn,static_cast<float>(b.config.rmsNormEps));
        // Pinned MNN 3.6.1 exposes forward SiLU but its UnaryGrad does not
        // implement UnaryOpOperation_SILU. Spell SiLU as x*sigmoid(x), which is
        // mathematically identical and uses BinaryOp + SIGMOID gradients that
        // are registered in this exact MNN commit.
        auto gate=linear(h,gw);
        auto ffv=(gate*_Sigmoid(gate))*linear(h,uw);
        x=x+linear(ffv,dw);
    }
    auto finalNorm=addP("final_norm.weight");
    x=rmsNorm(x,finalNorm,static_cast<float>(b.config.rmsNormEps));
    g.logits=_MatMul(x,embed,false,true); // tied LM head
    g.logits->setName("logits");

    auto onehot=_OneHot(g.targetInput,_Scalar<int32_t>(V),scalar(1.0f),scalar(0.0f),-1);
    // Stable log-softmax: logits - log(sum(exp(logits-max))) - max.
    auto mx=_ReduceMax(g.logits,{1},true);
    auto logsum=_Log(_ReduceSum(_Exp(g.logits-mx),{1},true))+mx;
    auto logp=g.logits-logsum;
    g.loss=_Negative(_ReduceMean(_ReduceSum(logp*onehot,{1},false),{},false));
    g.loss->setName("loss");
    return g;
}

void feedGraph(Graph& g,const Bundle& b) {
    auto* xp=g.tokenInput->writeMap<int32_t>();
    auto* yp=g.targetInput->writeMap<int32_t>();
    req(xp&&yp,"cannot map MNN graph inputs");
    for(int i=0;i<S;++i) { xp[i]=b.sampleTokens[i]; yp[i]=b.sampleTokens[i+1]; }
}

struct BackendCounts {
    int cpu=0, opencl=0, vulkan=0, other=0, callbacks=0;
};

void installProfiler(const std::shared_ptr<Executor>& exe, BackendCounts* counts) {
    exe->setCallBack(
        [](const std::vector<Tensor*>&, const OperatorInfo*) { return true; },
        [counts](const std::vector<Tensor*>& ts,const OperatorInfo*) {
            counts->callbacks++;
            std::set<int> seen;
            for(auto* t:ts) {
                if(!t) continue;
                auto* d=TensorUtils::getDescribeOrigin(t);
                auto* bn=d?d->getBackend():nullptr;
                if(!bn) continue;
                if(!seen.insert(static_cast<int>(bn->type())).second) continue;
                switch(bn->type()) {
                    case MNN_FORWARD_CPU: counts->cpu++; break;
                    case MNN_FORWARD_OPENCL: counts->opencl++; break;
                    case MNN_FORWARD_VULKAN: counts->vulkan++; break;
                    default: counts->other++; break;
                }
            }
            return true;
        });
}

std::shared_ptr<Executor> makeExecutor(MNNForwardType type,int threads,int gpuMode) {
    BackendConfig c;
    c.precision=BackendConfig::Precision_High;
    c.power=BackendConfig::Power_High;
    c.memory=BackendConfig::Memory_Normal;
    int n=(type==MNN_FORWARD_CPU)?threads:gpuMode;
    return Executor::newExecutor(type,c,n);
}

double relerr(double a,double b) { return std::abs(a-b)/std::max({1e-12,std::abs(a),std::abs(b)}); }

struct Parity {
    std::string backend;
    double loss=0, lossAbs=0;
    double gradNorm=0, gradNormRel=0;
    double maxLogitAbs=0, maxGradProbeAbs=0, maxGradProbeRel=0, maxAdamAbs=0, maxAdamRel=0;
    std::string worstGradSlot, worstAdamSlot;
    int worstGradIndex=-1, worstAdamIndex=-1;
    double worstGradRef=0, worstGradGot=0, worstAdamRef=0, worstAdamGot=0;
    BackendCounts counts;
    bool pass=false;
    std::string error;
};

void requireCompleteGradients(const Graph& g,const std::map<VARP,VARP>& gm,const std::string& label) {
    if(gm.size()==g.params.size()) return;
    std::ostringstream o;
    o<<label<<": MNN autograd incomplete: got "<<gm.size()<<"/"<<g.params.size()<<" parameter gradients; missing=[";
    bool first=true;
    for(const auto& kv:g.params) {
        if(gm.find(kv.second)!=gm.end()) continue;
        if(!first) o<<",";
        first=false;
        o<<kv.first;
    }
    o<<"]";
    throw std::runtime_error(o.str());
}

std::map<VARP,VARP> gradients(Graph& g) {
    OpGrad::init();
    markStage("gradients:enter");

    // Do not differentiate the numerically-stable logsumexp loss expression
    // through MNN 3.6.1. Its ReduceMaxGrad normalizes the max mask with a
    // global ReduceSum(mask), not along the requested reduction axis, so a
    // row-wise [S,V] max produces an incorrect backward seed.
    //
    // For mean cross entropy the exact analytic seed at logits is:
    //     dL/dlogits = (softmax(logits) - onehot(targets)) / S
    auto onehot=_OneHot(g.targetInput,_Scalar<int32_t>(V),scalar(1.0f),scalar(0.0f),-1);
    auto dlogits=(_Softmax(g.logits,1)-onehot)/scalar(static_cast<float>(S));

    // The pinned MNN GatherGrad is not correct for an embedding lookup:
    //   indices [S] -> reshape [S,1]
    //   backwardOutput [S,D] -> UNSQUEEZE(axis=0) -> [1,S,D]
    //   ScatterNd(indices, updates, shape) with no ADD reduction
    //
    // ScatterNd's own API/tests require updates [S,D] for indices [S,1], and
    // repeated token ids must be accumulated with BinaryOpOperation_ADD.
    // The embedding is tied, so its exact gradient is the sum of:
    //   (a) direct LM-head MatMul gradient, and
    //   (b) input embedding-lookup gradient scattered back by token id.
    //
    // Ask MNN autograd for all model parameters PLUS the lookup activation,
    // while blocking propagation through the Gather expression. This keeps the
    // correct direct LM-head contribution to the tied embedding, returns the
    // downstream dL/d(lookup), and prevents buggy GatherGrad from contributing.
    std::vector<VARP> targets;
    targets.reserve(g.params.size()+1);
    for(const auto& kv:g.params) targets.push_back(kv.second);
    targets.push_back(g.embeddingLookup);

    markStage("gradients:gradLinear:start");
    auto grads=OpGrad::gradLinear(g.logits,targets,{dlogits},{"embedding_lookup"});
    markStage("gradients:gradLinear:done");
    req(grads.size()==targets.size(),"dynamic parity: MNN gradLinear size mismatch");

    std::map<VARP,VARP> m;
    for(size_t i=0;i<g.params.size();++i) {
        if(grads[i].get()!=nullptr) m.emplace(targets[i],grads[i]);
    }

    auto lookupGrad=grads.back();
    req(lookupGrad.get()!=nullptr,"dynamic parity: embedding lookup gradient missing");

    auto embIt=g.params.find("tok_embeddings.weight");
    req(embIt!=g.params.end(),"dynamic parity: tied embedding parameter missing");
    auto directIt=m.find(embIt->second);
    req(directIt!=m.end(),"dynamic parity: direct LM-head embedding gradient missing");

    auto tokenIndex=_Reshape(g.tokenInput,{S,1});
    int32_t embShapeData[2]={V,D};
    auto embShape=_Const(embShapeData,{2},NHWC,halide_type_of<int32_t>());
    markStage("gradients:scatter_add:build:start");
    auto lookupWeightGrad=_ScatterNd(
        tokenIndex,
        lookupGrad,
        embShape,
        MNN::BinaryOpOperation_ADD
    );
    markStage("gradients:scatter_add:build:done");
    directIt->second=directIt->second+lookupWeightGrad;

    requireCompleteGradients(g,m,"dynamic parity");
    markStage("gradients:return");
    return m;
}

double hostGlobalNorm(const std::map<VARP,VARP>& gm) {
    long double sum=0;
    for(const auto& kv:gm) {
        const auto* info=kv.second->getInfo();
        req(info&&info->size>0,"gradient info missing");
        const float* p=kv.second->readMap<float>();
        req(p,"gradient readMap failed");
        for(size_t i=0;i<info->size;++i) { const long double x=p[i]; sum+=x*x; }
    }
    return std::sqrt(static_cast<double>(sum));
}

const rapidjson::Value& refGradient(const Bundle& b,const std::string& slot) {
    const auto& r=b.manifest["reference"];
    req(r.HasMember("gradient")&&r["gradient"].IsObject(),"reference gradient map missing");
    const auto& x=r["gradient"];
    req(x.HasMember(slot.c_str()),"reference gradient missing "+slot);
    return x[slot.c_str()];
}

const rapidjson::Value& refAdam(const Bundle& b,const std::string& slot) {
    const auto& r=b.manifest["reference"];
    req(r.HasMember("adamw_step1")&&r["adamw_step1"].IsObject(),"reference AdamW map missing");
    const auto& x=r["adamw_step1"];
    req(x.HasMember(slot.c_str()),"reference AdamW missing "+slot);
    return x[slot.c_str()];
}

VARP clippedGrad(VARP g, VARP coef) { return g*coef; }

struct UpdateExpressions {
    std::map<std::string,VARP> pnew;
    VARP norm;
};

UpdateExpressions oneStepExpressions(Graph& g,const std::map<VARP,VARP>& gm,const Bundle& b) {
    UpdateExpressions u;
    VARP sum=scalar(0.0f);
    for(const auto& kv:gm) sum=sum+_ReduceSum(_Square(kv.second),{},false);
    u.norm=_Sqrt(sum);
    auto coef=_Minimum(scalar(1.0f),scalar(1.0f)/(u.norm+scalar(1e-6f)));
    const float lr=static_cast<float>(b.adam.gateLr);
    const float b1=static_cast<float>(b.adam.beta1);
    const float b2=static_cast<float>(b.adam.beta2);
    const float eps=static_cast<float>(b.adam.eps);
    const float bc1=1.0f-b1, bc2=1.0f-b2;
    for(const auto& kv:g.params) {
        auto it=gm.find(kv.second); req(it!=gm.end(),"missing gradient "+kv.first);
        auto gg=it->second*coef;
        // Fresh moments are zero, so first-step expressions reduce cleanly but retain
        // exact AdamW bias correction semantics.
        auto m=(scalar(1.0f-b1))*gg;
        auto v=(scalar(1.0f-b2))*_Square(gg);
        auto denom=_Sqrt(v)/scalar(std::sqrt(bc2))+scalar(eps);
        auto adam=m/scalar(bc1)/denom;
        const float wd=static_cast<float>(b.adam.slotWeightDecay.at(kv.first));
        auto np=kv.second*scalar(1.0f-lr*wd)-scalar(lr)*adam;
        u.pnew.emplace(kv.first,np);
    }
    return u;
}

Parity dynamicParity(const Bundle& b,MNNForwardType type,int gpuMode) {
    Parity p; p.backend=typeName(type);
    markStage(p.backend + ":dynamic:enter");
    auto exe=makeExecutor(type,4,gpuMode);
    req(exe!=nullptr,"cannot create "+p.backend+" executor");
    ExecutorScope scope(exe);
    installProfiler(exe,&p.counts);
    markStage(p.backend + ":dynamic:build_graph:start");
    auto g=buildGraph(b); feedGraph(g,b);
    markStage(p.backend + ":dynamic:build_graph:done");

    markStage(p.backend + ":dynamic:gradients:start");
    auto gm=gradients(g);
    markStage(p.backend + ":dynamic:gradients:done");

    auto upd=oneStepExpressions(g,gm,b);
    markStage(p.backend + ":dynamic:updates_built");

    markStage(p.backend + ":dynamic:loss_read:start");
    const float* lp=g.loss->readMap<float>(); req(lp,"loss readMap failed");
    markStage(p.backend + ":dynamic:loss_read:done");
    p.loss=lp[0]; p.lossAbs=std::abs(p.loss-b.reference.loss);
    markStage(p.backend + ":dynamic:gradnorm:start");
    p.gradNorm=hostGlobalNorm(gm);
    markStage(p.backend + ":dynamic:gradnorm:done");
    p.gradNormRel=relerr(p.gradNorm,b.reference.globalGradNorm);

    markStage(p.backend + ":dynamic:logits_read:start");
    const float* logits=g.logits->readMap<float>(); req(logits,"logits readMap failed");
    markStage(p.backend + ":dynamic:logits_read:done");
    const auto& probes=b.manifest["reference"]["logit_probe"];
    for(const auto& q:probes.GetArray()) {
        int pos=q["position"].GetInt(), tok=q["token"].GetInt();
        double ref=q["value"].GetDouble();
        p.maxLogitAbs=std::max(p.maxLogitAbs,std::abs(static_cast<double>(logits[pos*V+tok])-ref));
    }

    for(const auto& kv:g.params) {
        const std::string& slot=kv.first;
        auto gi=gm.find(kv.second); req(gi!=gm.end(),"gradient map mismatch");
        const float* gd=gi->second->readMap<float>(); req(gd,"gradient map failed");
        const auto& rg=refGradient(b,slot);
        const auto& inds=rg["probe_indices"]; const auto& vals=rg["probe_values"];
        for(rapidjson::SizeType i=0;i<inds.Size();++i) {
            int idx=inds[i].GetInt(); double ref=vals[i].GetDouble(), got=gd[idx];
            const double ae=std::abs(got-ref);
            const double re=relerr(got,ref);
            if(ae>p.maxGradProbeAbs) {
                p.maxGradProbeAbs=ae;
                p.worstGradSlot=slot;
                p.worstGradIndex=idx;
                p.worstGradRef=ref;
                p.worstGradGot=got;
            }
            p.maxGradProbeRel=std::max(p.maxGradProbeRel,re);
        }

        const float* nd=upd.pnew.at(slot)->readMap<float>(); req(nd,"AdamW update map failed");
        const auto& ra=refAdam(b,slot);
        const auto& ai=ra["probe_indices"]; const auto& av=ra["after"];
        for(rapidjson::SizeType i=0;i<ai.Size();++i) {
            int idx=ai[i].GetInt(); double ref=av[i].GetDouble(), got=nd[idx];
            const double ae=std::abs(got-ref);
            const double re=relerr(got,ref);
            if(ae>p.maxAdamAbs) {
                p.maxAdamAbs=ae;
                p.worstAdamSlot=slot;
                p.worstAdamIndex=idx;
                p.worstAdamRef=ref;
                p.worstAdamGot=got;
            }
            p.maxAdamRel=std::max(p.maxAdamRel,re);
        }
    }

    // CPU parity is intentionally strict; GPU still must remain close enough to
    // the PyTorch FP32 reference to be scientifically interchangeable at a stage boundary.
    markStage(p.backend + ":dynamic:probe_reads:done");
    p.pass=std::isfinite(p.loss)&&std::isfinite(p.gradNorm)
        && p.lossAbs<=2e-3 && p.maxLogitAbs<=5e-3
        && p.gradNormRel<=2e-2 && p.maxGradProbeAbs<=5e-3
        && p.maxAdamAbs<=5e-4;
    return p;
}

Parity safeDynamicParity(const Bundle& b,MNNForwardType type,int gpuMode) {
    try {
        return dynamicParity(b,type,gpuMode);
    } catch(const std::exception& e) {
        Parity p;
        p.backend=typeName(type);
        p.pass=false;
        p.error=e.what();
        return p;
    }
}

std::string parityJson(const Parity& p) {
    std::ostringstream o;
    o<<"{\"backend\":\""<<p.backend<<"\",\"pass\":"<<(p.pass?"true":"false")
     <<",\"loss\":"<<p.loss<<",\"loss_abs_error\":"<<p.lossAbs
     <<",\"global_grad_norm\":"<<p.gradNorm<<",\"grad_norm_rel_error\":"<<p.gradNormRel
     <<",\"max_logit_abs_error\":"<<p.maxLogitAbs
     <<",\"max_grad_probe_abs_error\":"<<p.maxGradProbeAbs
     <<",\"max_grad_probe_rel_error\":"<<p.maxGradProbeRel
     <<",\"max_adamw_probe_abs_error\":"<<p.maxAdamAbs
     <<",\"max_adamw_probe_rel_error\":"<<p.maxAdamRel
     <<",\"worst_grad\":{\"slot\":\""<<jsonEscape(p.worstGradSlot)<<"\",\"index\":"<<p.worstGradIndex
     <<",\"ref\":"<<p.worstGradRef<<",\"got\":"<<p.worstGradGot<<"}"
     <<",\"worst_adamw\":{\"slot\":\""<<jsonEscape(p.worstAdamSlot)<<"\",\"index\":"<<p.worstAdamIndex
     <<",\"ref\":"<<p.worstAdamRef<<",\"got\":"<<p.worstAdamGot<<"}"
     <<",\"backend_counts\":{\"cpu\":"<<p.counts.cpu<<",\"opencl\":"<<p.counts.opencl
     <<",\"vulkan\":"<<p.counts.vulkan<<",\"other\":"<<p.counts.other
     <<",\"callbacks\":"<<p.counts.callbacks<<"},\"error\":\""<<jsonEscape(p.error)<<"\"}";
    return o.str();
}

enum class StaticProbeKind { Logit, Gradient, AdamW };

struct StaticProbeSpec {
    std::string name;
    StaticProbeKind kind;
    std::string slot;
    int index=0;
    double ref=0.0;
};

struct StaticParityBuild {
    std::string path;
    std::vector<StaticProbeSpec> probes;
};

VARP scalarProbe(VARP x,int index,const std::string& name) {
    auto flat=_Reshape(x,{-1});
    auto v=_Gather(flat,_Scalar<int32_t>(index));
    v->setName(name);
    return v;
}

StaticParityBuild buildStaticParityModel(const Bundle& b,const std::string& path) {
    // Build the exact one-step verification graph on CPU, then serialize it.
    // Only compact outputs are retained: loss, global grad norm, selected
    // logits, selected gradients, and selected fresh-state AdamW values.
    // This preserves the same scientific checks as dynamic parity without
    // host-mapping all 19M gradient elements on OpenCL.
    auto exe=makeExecutor(MNN_FORWARD_CPU,4,0);
    ExecutorScope scope(exe);

    auto g=buildGraph(b);
    auto gm=gradients(g);
    auto upd=oneStepExpressions(g,gm,b);

    g.loss->setName("parity.loss");
    upd.norm->setName("parity.global_grad_norm");

    std::vector<VARP> outputs={g.loss,upd.norm};
    StaticParityBuild out;
    out.path=path;

    int serial=0;
    const auto& logitProbes=b.manifest["reference"]["logit_probe"];
    for(const auto& q:logitProbes.GetArray()) {
        const int pos=q["position"].GetInt();
        const int tok=q["token"].GetInt();
        const int index=pos*V+tok;
        const std::string name="parity.logit."+std::to_string(serial++);
        outputs.push_back(scalarProbe(g.logits,index,name));
        out.probes.push_back({name,StaticProbeKind::Logit,"logits",index,q["value"].GetDouble()});
    }

    serial=0;
    for(const auto& pk:g.params) {
        const auto& rg=refGradient(b,pk.first);
        const auto& inds=rg["probe_indices"];
        const auto& vals=rg["probe_values"];
        auto grad=gm.at(pk.second);
        for(rapidjson::SizeType i=0;i<inds.Size();++i) {
            const int index=inds[i].GetInt();
            const std::string name="parity.grad."+std::to_string(serial++);
            outputs.push_back(scalarProbe(grad,index,name));
            out.probes.push_back({name,StaticProbeKind::Gradient,pk.first,index,vals[i].GetDouble()});
        }
    }

    serial=0;
    for(const auto& pk:g.params) {
        const auto& ra=refAdam(b,pk.first);
        const auto& inds=ra["probe_indices"];
        const auto& vals=ra["after"];
        auto pnew=upd.pnew.at(pk.first);
        for(rapidjson::SizeType i=0;i<inds.Size();++i) {
            const int index=inds[i].GetInt();
            const std::string name="parity.adam."+std::to_string(serial++);
            outputs.push_back(scalarProbe(pnew,index,name));
            out.probes.push_back({name,StaticProbeKind::AdamW,pk.first,index,vals[i].GetDouble()});
        }
    }

    markStage("static_parity:serialize:start");
    Variable::save(outputs,path.c_str());
    markStage("static_parity:serialize:done");

    std::ifstream fs(path,std::ios::binary|std::ios::ate);
    req(fs&&fs.tellg()>0,"static parity model serialization failed");
    return out;
}

struct StaticParityResult {
    std::string backend;
    bool available=false;
    bool finite=false;
    bool pass=false;
    double loss=0.0;
    double lossAbs=0.0;
    double gradNorm=0.0;
    double gradNormRel=0.0;
    double maxLogitAbs=0.0;
    double maxGradAbs=0.0;
    double maxGradRel=0.0;
    double maxAdamAbs=0.0;
    double maxAdamRel=0.0;
    std::string worstGradSlot;
    std::string worstAdamSlot;
    int worstGradIndex=-1;
    int worstAdamIndex=-1;
    double worstGradRef=0.0;
    double worstGradGot=0.0;
    double worstAdamRef=0.0;
    double worstAdamGot=0.0;
    double sessionMemoryMb=0.0;
    BackendCounts counts;
    std::string error;
};

StaticParityResult staticParity(
    const Bundle& b,
    const StaticParityBuild& spec,
    MNNForwardType type,
    int gpuMode
) {
    StaticParityResult r;
    r.backend=typeName(type);

    markStage(r.backend+":static_parity:interpreter:start");
    std::shared_ptr<Interpreter> net(
        Interpreter::createFromFile(spec.path.c_str()),Interpreter::destroy);
    req(net!=nullptr,"static parity Interpreter failed on "+r.backend);
    markStage(r.backend+":static_parity:interpreter:done");

    BackendConfig bc;
    bc.precision=BackendConfig::Precision_High;
    bc.power=BackendConfig::Power_High;
    bc.memory=(type==MNN_FORWARD_OPENCL)
        ? BackendConfig::Memory_Low
        : BackendConfig::Memory_Normal;

    ScheduleConfig cfg;
    cfg.type=type;
    cfg.backendConfig=&bc;
    if(type==MNN_FORWARD_CPU) cfg.numThread=4;
    else cfg.mode=gpuMode;

    // Variable::save() serializes tensor names but does NOT populate
    // Net::outputName. MNN Session auto-detection therefore exposes only
    // graph-leaf tensors as outputs. parity.global_grad_norm is intentionally
    // consumed by the AdamW probe expressions, so it is not a leaf. The
    // documented ScheduleConfig::saveTensors mechanism is exactly for this:
    // retain named intermediates as session outputs. Keep this list compact so
    // MNN may still reuse every other activation/gradient buffer.
    cfg.saveTensors.push_back("parity.loss");
    cfg.saveTensors.push_back("parity.global_grad_norm");
    for(const auto& p:spec.probes) cfg.saveTensors.push_back(p.name);

    markStage(r.backend+":static_parity:create_session:start");
    auto* session=net->createSession(cfg);
    req(session!=nullptr,"static parity session failed on "+r.backend);
    markStage(r.backend+":static_parity:create_session:done");
    r.available=true;

    net->getSessionInfo(session,MNN::Interpreter::MEMORY,&r.sessionMemoryMb);

    auto* ti=net->getSessionInput(session,"tokens");
    auto* yi=net->getSessionInput(session,"targets");
    auto* lo=net->getSessionOutput(session,"parity.loss");
    auto* gn=net->getSessionOutput(session,"parity.global_grad_norm");
    if(!(ti&&yi&&lo&&gn)) {
        std::ostringstream e;
        e<<"static parity IO contract missing on "<<r.backend
         <<": tokens="<<(ti?"yes":"NO")
         <<", targets="<<(yi?"yes":"NO")
         <<", loss="<<(lo?"yes":"NO")
         <<", global_grad_norm="<<(gn?"yes":"NO")
         <<"; inputs=[";
        bool first=true;
        for(const auto& kv:net->getSessionInputAll(session)) {
            if(!first)e<<",";
            first=false;
            e<<kv.first;
        }
        e<<"]; outputs=[";
        first=true;
        for(const auto& kv:net->getSessionOutputAll(session)) {
            if(!first)e<<",";
            first=false;
            e<<kv.first;
        }
        e<<"]";
        throw std::runtime_error(e.str());
    }

    {
        Tensor th(ti,Tensor::CAFFE);
        Tensor yh(yi,Tensor::CAFFE);
        auto* p=th.host<int32_t>();
        auto* q=yh.host<int32_t>();
        req(p&&q,"static parity host input allocation failed");
        for(int i=0;i<S;++i){p[i]=b.sampleTokens[i];q[i]=b.sampleTokens[i+1];}
        ti->copyFromHostTensor(&th);
        yi->copyFromHostTensor(&yh);
    }

    // No more session creation/resizing is needed for this read-only parity
    // model. MNN documents releaseModel() specifically to drop the interpreter
    // model buffer after session creation and save roughly the model-file size.
    net->releaseModel();

    auto before=[](const std::vector<Tensor*>&,const OperatorInfo*){return true;};
    auto after=[&](const std::vector<Tensor*>& ts,const OperatorInfo*) {
        std::set<int> types;
        for(auto* t:ts) {
            auto* bn=t?net->getBackend(session,t):nullptr;
            if(bn) types.insert(static_cast<int>(bn->type()));
        }
        for(int x:types) {
            if(x==MNN_FORWARD_CPU) r.counts.cpu++;
            else if(x==MNN_FORWARD_OPENCL) r.counts.opencl++;
            else if(x==MNN_FORWARD_VULKAN) r.counts.vulkan++;
            else r.counts.other++;
        }
        r.counts.callbacks++;
        return true;
    };

    markStage(r.backend+":static_parity:run:start");
    auto ec=net->runSessionWithCallBackInfo(session,before,after,true);
    req(ec==NO_ERROR,"static parity run failed on "+r.backend);
    markStage(r.backend+":static_parity:run:done");

    auto readScalar=[&](Tensor* t,const std::string& label)->double {
        req(t!=nullptr,"static parity output missing: "+label);
        Tensor host(t,Tensor::CAFFE);
        t->copyToHostTensor(&host);
        auto* p=host.host<float>();
        req(p!=nullptr,"static parity host read failed: "+label);
        return static_cast<double>(p[0]);
    };

    markStage(r.backend+":static_parity:small_outputs:start");
    r.loss=readScalar(lo,"loss");
    r.gradNorm=readScalar(gn,"global_grad_norm");
    r.lossAbs=std::abs(r.loss-b.reference.loss);
    r.gradNormRel=relerr(r.gradNorm,b.reference.globalGradNorm);

    for(const auto& p:spec.probes) {
        auto* t=net->getSessionOutput(session,p.name.c_str());
        const double got=readScalar(t,p.name);
        const double ae=std::abs(got-p.ref);
        const double re=relerr(got,p.ref);
        switch(p.kind) {
            case StaticProbeKind::Logit:
                r.maxLogitAbs=std::max(r.maxLogitAbs,ae);
                break;
            case StaticProbeKind::Gradient:
                if(ae>r.maxGradAbs) {
                    r.maxGradAbs=ae;
                    r.worstGradSlot=p.slot;
                    r.worstGradIndex=p.index;
                    r.worstGradRef=p.ref;
                    r.worstGradGot=got;
                }
                r.maxGradRel=std::max(r.maxGradRel,re);
                break;
            case StaticProbeKind::AdamW:
                if(ae>r.maxAdamAbs) {
                    r.maxAdamAbs=ae;
                    r.worstAdamSlot=p.slot;
                    r.worstAdamIndex=p.index;
                    r.worstAdamRef=p.ref;
                    r.worstAdamGot=got;
                }
                r.maxAdamRel=std::max(r.maxAdamRel,re);
                break;
        }
    }
    markStage(r.backend+":static_parity:small_outputs:done");

    r.finite=
        std::isfinite(r.loss)&&std::isfinite(r.gradNorm)&&
        std::isfinite(r.maxLogitAbs)&&std::isfinite(r.maxGradAbs)&&
        std::isfinite(r.maxAdamAbs);

    const bool backendOk=(type!=MNN_FORWARD_OPENCL)||r.counts.opencl>0;
    r.pass=
        r.finite&&backendOk&&
        r.lossAbs<=2e-3&&
        r.maxLogitAbs<=5e-3&&
        r.gradNormRel<=2e-2&&
        r.maxGradAbs<=5e-3&&
        r.maxAdamAbs<=5e-4;

    net->releaseSession(session);
    return r;
}

StaticParityResult safeStaticParity(
    const Bundle& b,
    const StaticParityBuild& spec,
    MNNForwardType type,
    int gpuMode
) {
    try {
        return staticParity(b,spec,type,gpuMode);
    } catch(const std::exception& e) {
        StaticParityResult r;
        r.backend=typeName(type);
        r.error=e.what();
        return r;
    }
}

std::string staticParityJson(const StaticParityResult& r) {
    std::ostringstream o;
    o<<"{\"backend\":\""<<r.backend<<"\",\"available\":"<<(r.available?"true":"false")
     <<",\"finite\":"<<(r.finite?"true":"false")
     <<",\"pass\":"<<(r.pass?"true":"false")
     <<",\"session_memory_mb\":"<<r.sessionMemoryMb
     <<",\"loss\":"<<r.loss
     <<",\"loss_abs_error\":"<<r.lossAbs
     <<",\"global_grad_norm\":"<<r.gradNorm
     <<",\"grad_norm_rel_error\":"<<r.gradNormRel
     <<",\"max_logit_abs_error\":"<<r.maxLogitAbs
     <<",\"max_grad_probe_abs_error\":"<<r.maxGradAbs
     <<",\"max_grad_probe_rel_error\":"<<r.maxGradRel
     <<",\"max_adamw_probe_abs_error\":"<<r.maxAdamAbs
     <<",\"max_adamw_probe_rel_error\":"<<r.maxAdamRel
     <<",\"worst_grad\":{\"slot\":\""<<jsonEscape(r.worstGradSlot)
     <<"\",\"index\":"<<r.worstGradIndex
     <<",\"ref\":"<<r.worstGradRef<<",\"got\":"<<r.worstGradGot<<"}"
     <<",\"worst_adamw\":{\"slot\":\""<<jsonEscape(r.worstAdamSlot)
     <<"\",\"index\":"<<r.worstAdamIndex
     <<",\"ref\":"<<r.worstAdamRef<<",\"got\":"<<r.worstAdamGot<<"}"
     <<",\"backend_counts\":{\"cpu\":"<<r.counts.cpu
     <<",\"opencl\":"<<r.counts.opencl
     <<",\"vulkan\":"<<r.counts.vulkan
     <<",\"other\":"<<r.counts.other
     <<",\"callbacks\":"<<r.counts.callbacks<<"}"
     <<",\"error\":\""<<jsonEscape(r.error)<<"\"}";
    return o.str();
}

struct StaticBuild {
    std::string path;
};

StaticBuild buildStaticAdamWModel(const Bundle& b,const std::string& path) {
    // Build on CPU; the serialized model is backend-neutral and is later loaded
    // independently by CPU/OpenCL/Vulkan sessions.
    auto exe=makeExecutor(MNN_FORWARD_CPU,4,0); ExecutorScope scope(exe);
    auto g=buildGraph(b);
    // Reuse the exact analytic cross-entropy seed used by dynamic parity.
    // Static training must not reintroduce MNN 3.6.1's ReduceMaxGrad defect.
    auto gm=gradients(g);

    VARP sum=scalar(0.0f);
    for(auto& kv:gm) sum=sum+_ReduceSum(_Square(kv.second),{},false);
    auto norm=_Sqrt(sum); norm->setName("global_grad_norm");
    auto coef=_Minimum(scalar(1.0f),scalar(1.0f)/(norm+scalar(1e-6f)));

    const float lr=static_cast<float>(b.adam.gateLr);
    const float b1=static_cast<float>(b.adam.beta1), b2=static_cast<float>(b.adam.beta2);
    const float eps=static_cast<float>(b.adam.eps);

    std::vector<VARP> oldState,newState;
    // Pow states contain beta^step for the step being applied; start at beta^1.
    auto b1pow=_TrainableParam(b1,{},NHWC); b1pow->setName("adamw.beta1_pow");
    auto b2pow=_TrainableParam(b2,{},NHWC); b2pow->setName("adamw.beta2_pow");

    for(auto& pk:g.params) {
        const auto& td=b.tensor(pk.first);
        auto m=_TrainableParam(0.0f,td.shape,NHWC); m->setName("adamw.m."+pk.first);
        auto v=_TrainableParam(0.0f,td.shape,NHWC); v->setName("adamw.v."+pk.first);
        auto gg=gm.at(pk.second)*coef;
        auto mn=scalar(b1)*m+scalar(1.0f-b1)*gg;
        auto vn=scalar(b2)*v+scalar(1.0f-b2)*_Square(gg);
        auto denom=_Sqrt(vn)/_Sqrt(scalar(1.0f)-b2pow)+scalar(eps);
        auto stepSize=scalar(lr)/(scalar(1.0f)-b1pow);
        const float wd=static_cast<float>(b.adam.slotWeightDecay.at(pk.first));
        auto pn=pk.second*scalar(1.0f-lr*wd)-stepSize*mn/denom;
        pn->setName("update."+pk.first);
        mn->setName("update.adamw.m."+pk.first);
        vn->setName("update.adamw.v."+pk.first);
        oldState.insert(oldState.end(),{pk.second,m,v});
        newState.insert(newState.end(),{pn,mn,vn});
    }
    auto b1n=b1pow*scalar(b1); b1n->setName("update.adamw.beta1_pow");
    auto b2n=b2pow*scalar(b2); b2n->setName("update.adamw.beta2_pow");
    oldState.insert(oldState.end(),{b1pow,b2pow}); newState.insert(newState.end(),{b1n,b2n});

    g.loss->setName("loss"); norm->setName("global_grad_norm");
    MNN::Train::ParameterOptimizer::makeLoopModel(path.c_str(),{g.loss,norm},{oldState,newState});
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    req(f&&f.tellg()>0,"static train model serialization failed");
    return {path};
}

struct Bench {
    std::string backend;
    bool available=false, finite=false;
    double firstLoss=0,lastLoss=0,seconds=0,tokps=0;
    int steps=0,cpuOps=0,gpuOps=0,otherOps=0;
    std::string checkpointPath;
    bool checkpointReloadOk=false;
    std::string error;
};

Bench benchStatic(const Bundle& b,const std::string& baseModel,const std::string& workDir,
                  MNNForwardType type,int gpuMode,int steps) {
    Bench out; out.backend=typeName(type);
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(baseModel.c_str()),Interpreter::destroy);
    if(!net) return out;
    BackendConfig bc; bc.precision=BackendConfig::Precision_High; bc.power=BackendConfig::Power_High;
    bc.memory=(type==MNN_FORWARD_OPENCL)?BackendConfig::Memory_Low:BackendConfig::Memory_Normal;
    ScheduleConfig cfg; cfg.type=type; cfg.backendConfig=&bc;
    if(type==MNN_FORWARD_CPU) cfg.numThread=4;
    else cfg.mode=gpuMode;
    auto* session=net->createSession(cfg);
    if(!session) return out;
    out.available=true;
    auto* ti=net->getSessionInput(session,"tokens");
    auto* yi=net->getSessionInput(session,"targets");
    auto* lo=net->getSessionOutput(session,"loss");
    req(ti&&yi&&lo,"static training model input/output names missing");
    {
        Tensor th(ti,Tensor::CAFFE); auto* p=th.host<int32_t>(); req(p,"token host tensor");
        Tensor yh(yi,Tensor::CAFFE); auto* q=yh.host<int32_t>(); req(q,"target host tensor");
        for(int i=0;i<S;++i){p[i]=b.sampleTokens[i];q[i]=b.sampleTokens[i+1];}
        ti->copyFromHostTensor(&th); yi->copyFromHostTensor(&yh);
    }

    BackendCounts counts;
    auto before=[](const std::vector<Tensor*>&,const OperatorInfo*){return true;};
    auto after=[&](const std::vector<Tensor*>& ts,const OperatorInfo*) {
        std::set<int> types;
        for(auto* t:ts) {
            auto* bn=t?net->getBackend(session,t):nullptr;
            if(bn) types.insert(static_cast<int>(bn->type()));
        }
        for(int x:types) {
            if(x==MNN_FORWARD_CPU) counts.cpu++;
            else if(x==MNN_FORWARD_OPENCL||x==MNN_FORWARD_VULKAN) counts.opencl++;
            else counts.other++;
        }
        return true;
    };

    // Warm-up/compile only. Do not include it in sustained timing.
    auto ec=net->runSessionWithCallBackInfo(session,before,after,true);
    req(ec==NO_ERROR,"static warmup failed on "+out.backend);
    Tensor lossHost(lo,Tensor::CAFFE); lo->copyToHostTensor(&lossHost);
    out.firstLoss=lossHost.host<float>()[0];

    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<steps;++i) {
        auto e=net->runSession(session);
        req(e==NO_ERROR,"static training step failed on "+out.backend);
    }
    auto t1=std::chrono::steady_clock::now();
    lo->copyToHostTensor(&lossHost);
    out.lastLoss=lossHost.host<float>()[0];
    out.seconds=std::chrono::duration<double>(t1-t0).count();
    out.steps=steps;
    out.tokps=(steps*S)/std::max(1e-9,out.seconds);
    out.finite=std::isfinite(out.firstLoss)&&std::isfinite(out.lastLoss);
    out.cpuOps=counts.cpu; out.gpuOps=counts.opencl; out.otherOps=counts.other;

    // Persist the trained session atomically. MNN officially exposes this path.
    auto ue=net->updateSessionToModel(session);
    req(ue==NO_ERROR,"updateSessionToModel failed");
    auto mb=net->getModelBuffer();
    req(mb.first&&mb.second>0,"getModelBuffer returned empty model");
    out.checkpointPath=workDir+"/gate-"+out.backend+".mnn";
    atomicWrite(out.checkpointPath,mb.first,mb.second);

    // Prove the persisted checkpoint is structurally reloadable, not merely writable.
    {
        std::shared_ptr<Interpreter> reloaded(
            Interpreter::createFromFile(out.checkpointPath.c_str()), Interpreter::destroy);
        req(reloaded!=nullptr,"checkpoint reload Interpreter failed on "+out.backend);
        auto* rs=reloaded->createSession(cfg);
        req(rs!=nullptr,"checkpoint reload session failed on "+out.backend);
        auto* rti=reloaded->getSessionInput(rs,"tokens");
        auto* ryi=reloaded->getSessionInput(rs,"targets");
        auto* rlo=reloaded->getSessionOutput(rs,"loss");
        req(rti&&ryi&&rlo,"checkpoint reload IO contract failed on "+out.backend);
        reloaded->releaseSession(rs);
        out.checkpointReloadOk=true;
    }

    net->releaseSession(session);
    return out;
}

Bench safeBenchStatic(const Bundle& b,const std::string& baseModel,const std::string& workDir,
                     MNNForwardType type,int gpuMode,int steps) {
    try {
        return benchStatic(b,baseModel,workDir,type,gpuMode,steps);
    } catch(const std::exception& e) {
        Bench out;
        out.backend=typeName(type);
        out.error=e.what();
        return out;
    }
}

std::string benchJson(const Bench& b) {
    std::ostringstream o;
    o<<"{\"backend\":\""<<b.backend<<"\",\"available\":"<<(b.available?"true":"false")
      <<",\"finite\":"<<(b.finite?"true":"false")<<",\"steps\":"<<b.steps
      <<",\"first_loss\":"<<b.firstLoss<<",\"last_loss\":"<<b.lastLoss
      <<",\"seconds\":"<<b.seconds<<",\"target_tokens_per_second\":"<<b.tokps
      <<",\"profile\":{\"cpu_backend_hits\":"<<b.cpuOps<<",\"gpu_backend_hits\":"<<b.gpuOps
      <<",\"other_backend_hits\":"<<b.otherOps<<"},\"checkpoint\":\""<<jsonEscape(b.checkpointPath)<<"\""
      <<",\"checkpoint_reload_ok\":"<<(b.checkpointReloadOk?"true":"false")
      <<",\"error\":\""<<jsonEscape(b.error)<<"\"}";
    return o.str();
}

// Native OpenCL visibility probe, independent of MNN scheduling.
std::string openClProbe() {
    const char* libs[]={"libOpenCL.so","libGLES_mali.so","libmali.so"};
    void* h=nullptr; std::string lib;
    for(auto* n:libs){h=dlopen(n,RTLD_NOW|RTLD_LOCAL);if(h){lib=n;break;}}
    if(!h) return "{\"library_visible\":false}";
    auto pPlatforms=reinterpret_cast<decltype(&clGetPlatformIDs)>(dlsym(h,"clGetPlatformIDs"));
    auto pDevices=reinterpret_cast<decltype(&clGetDeviceIDs)>(dlsym(h,"clGetDeviceIDs"));
    auto pInfo=reinterpret_cast<decltype(&clGetDeviceInfo)>(dlsym(h,"clGetDeviceInfo"));
    if(!pPlatforms||!pDevices||!pInfo){dlclose(h);return "{\"library_visible\":true,\"symbols\":false}";}
    cl_uint np=0; cl_int e=pPlatforms(0,nullptr,&np);
    if(e!=CL_SUCCESS||np==0){dlclose(h);return "{\"library_visible\":true,\"symbols\":true,\"platforms\":0}";}
    std::vector<cl_platform_id> ps(np);pPlatforms(np,ps.data(),nullptr);
    cl_device_id dev=nullptr;
    for(auto p:ps){cl_uint nd=0;if(pDevices(p,CL_DEVICE_TYPE_GPU,1,&dev,&nd)==CL_SUCCESS&&nd>0)break;}
    if(!dev){dlclose(h);return "{\"library_visible\":true,\"symbols\":true,\"gpu\":false}";}
    char name[256]={0},vendor[256]={0},version[256]={0};
    cl_uint cu=0,mhz=0; cl_ulong mem=0;
    pInfo(dev,CL_DEVICE_NAME,sizeof(name),name,nullptr);
    pInfo(dev,CL_DEVICE_VENDOR,sizeof(vendor),vendor,nullptr);
    pInfo(dev,CL_DEVICE_VERSION,sizeof(version),version,nullptr);
    pInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cu),&cu,nullptr);
    pInfo(dev,CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(mhz),&mhz,nullptr);
    pInfo(dev,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(mem),&mem,nullptr);
    std::ostringstream o;
    o<<"{\"library_visible\":true,\"library\":\""<<jsonEscape(lib)<<"\",\"symbols\":true,\"gpu\":true"
     <<",\"name\":\""<<jsonEscape(name)<<"\",\"vendor\":\""<<jsonEscape(vendor)
     <<"\",\"version\":\""<<jsonEscape(version)<<"\",\"compute_units\":"<<cu
     <<",\"max_clock_mhz\":"<<mhz<<",\"global_mem_bytes\":"<<static_cast<unsigned long long>(mem)<<"}";
    dlclose(h);return o.str();
}

} // namespace

std::string probeBackendsJson() {
    try {
        auto cpu=makeExecutor(MNN_FORWARD_CPU,4,0);
        auto cl=makeExecutor(MNN_FORWARD_OPENCL,1,MNN_GPU_TUNING_FAST|MNN_GPU_MEMORY_IMAGE);
        auto vk=makeExecutor(MNN_FORWARD_VULKAN,1,MNN_GPU_TUNING_WIDE);
        std::ostringstream o;
        o<<"{\"status\":\"PASS\",\"mnn_commit\":\""<<ANDROID_TRAINER_MNN_COMMIT
         <<"\",\"opencl\":"<<openClProbe()
         <<",\"executors\":{\"cpu\":"<<(cpu?"true":"false")
         <<",\"opencl\":"<<(cl?"true":"false")<<",\"vulkan_buffer\":"<<(vk?"true":"false")<<"}}";
        markStage("run:success");
        return o.str();
    } catch(const std::exception& e) {
        markStage(std::string("run:exception:")+e.what());
        return std::string("{\"status\":\"FAIL\",\"error\":\"")+jsonEscape(e.what())+"\"}";
    }
}

std::string validateBundleJson(const std::string& dir) {
    try {
        auto b=Bundle::load(dir);
        std::ostringstream o;
        o<<"{\"status\":\"PASS\",\"schema\":\""<<b.schema<<"\",\"params\":"<<b.parameterCount
         <<",\"tensors\":"<<b.tensors.size()<<",\"checkpoint_sha256\":\""<<b.checkpointSha256
         <<"\",\"rope_style\":\""<<b.ropeStyle<<"\",\"reference_loss\":"<<b.reference.loss<<"}";
        return o.str();
    } catch(const std::exception& e) {
        return std::string("{\"status\":\"FAIL\",\"error\":\"")+jsonEscape(e.what())+"\"}";
    }
}

std::string runModel0001GateJson(const std::string& dir,const std::string& workDir,float thermalHeadroom) {
    gStagePath=workDir+"/last_native_stage.txt";
    markStage("run:enter");
    try {
        auto b=Bundle::load(dir);
        markStage("run:bundle_loaded");

        // 1) Reference parity on MNN CPU. If the Python source did not expose
        // the RoPE layout clearly, empirically select between the two standard
        // layouts using the frozen PyTorch reference. Nothing is guessed silently.
        Parity cpu;
        std::string ropeEvidence;
        if (b.ropeStyle == "auto") {
            b.ropeStyle = "half_split";
            Parity half = dynamicParity(b,MNN_FORWARD_CPU,0);
            if (half.pass) {
                cpu = half;
                ropeEvidence = "auto->half_split";
            } else {
                b.ropeStyle = "interleaved";
                Parity inter = dynamicParity(b,MNN_FORWARD_CPU,0);
                if (inter.pass) {
                    cpu = inter;
                    ropeEvidence = "auto->interleaved";
                } else {
                    return std::string("{\"status\":\"FAIL_CPU_PARITY\",\"reason\":\"neither supported RoPE layout matches PyTorch\",\"half_split\":")+
                        parityJson(half)+",\"interleaved\":"+parityJson(inter)+"}";
                }
            }
        } else {
            ropeEvidence = "declared->" + b.ropeStyle;
            markStage("run:cpu_parity:start");
            cpu = dynamicParity(b,MNN_FORWARD_CPU,0);
            markStage("run:cpu_parity:done");
        }
        if(!cpu.pass) {
            return std::string("{\"status\":\"FAIL_CPU_PARITY\",\"thermal_headroom_start\":")+
                std::to_string(thermalHeadroom)+",\"cpu_parity\":"+parityJson(cpu)+"}";
        }

        // 2) The phone has already proven that MNN Express dynamic OpenCL
        // parity is the wrong execution architecture for this gate: Android
        // killed the process with REASON_LOW_MEMORY exactly at
        // OPENCL:dynamic:gradnorm:start, where hostGlobalNorm() materializes
        // every gradient tensor on the host. Keep dynamic autograd as the CPU
        // oracle, but move GPU correctness to a serialized Session graph so MNN
        // can plan/reuse all non-output intermediate memory.
        Parity clDynamic;
        clDynamic.backend="OPENCL";
        clDynamic.pass=false;
        clDynamic.error="skipped: proven Android LOW_MEMORY during dynamic OpenCL gradient materialization";
        Parity vk;
        vk.backend="VULKAN";
        vk.pass=false;
        vk.error="disabled_by_project_policy";

        // 3) Serialize a read-only one-step parity graph with ONLY compact
        // outputs: loss, global grad norm, the locked logit probes, the locked
        // gradient probes, and the locked fresh-state AdamW probes. This checks
        // the same semantics as CPU dynamic parity without copying 74 complete
        // gradients back from GPU.
        const std::string parityModel=workDir+"/model0001-gate-static-parity.mnn";
        markStage("run:static_parity_build:start");
        auto paritySpec=buildStaticParityModel(b,parityModel);
        markStage("run:static_parity_build:done");

        markStage("run:cpu_static_parity:start");
        auto cpuStatic=safeStaticParity(b,paritySpec,MNN_FORWARD_CPU,0);
        markStage("run:cpu_static_parity:done");
        if(!cpuStatic.pass) {
            return std::string("{\"status\":\"FAIL_CPU_STATIC_PARITY\",\"thermal_headroom_start\":")+
                std::to_string(thermalHeadroom)+
                ",\"cpu_dynamic\":"+parityJson(cpu)+
                ",\"cpu_static\":"+staticParityJson(cpuStatic)+"}";
        }

        markStage("run:opencl_static_parity:start");
        auto clStatic=safeStaticParity(
            b,paritySpec,MNN_FORWARD_OPENCL,
            MNN_GPU_TUNING_FAST|MNN_GPU_MEMORY_IMAGE);
        markStage("run:opencl_static_parity:done");
        if(!clStatic.pass) {
            return std::string("{\"status\":\"FAIL_OPENCL_STATIC_PARITY\",\"thermal_headroom_start\":")+
                std::to_string(thermalHeadroom)+
                ",\"cpu_dynamic\":"+parityJson(cpu)+
                ",\"cpu_static\":"+staticParityJson(cpuStatic)+
                ",\"opencl_static\":"+staticParityJson(clStatic)+
                ",\"opencl_runtime\":"+openClProbe()+"}";
        }

        // 4) Only after exact static OpenCL parity passes, build the stateful
        // AdamW loop and measure sustained CPU vs OpenCL training. This graph
        // is backend-neutral and each Session gets its own independent state.
        const std::string base=workDir+"/model0001-gate-static-base.mnn";
        markStage("run:static_train_build:start");
        buildStaticAdamWModel(b,base);
        markStage("run:static_train_build:done");

        const int steps=20;
        markStage("run:cpu_static_train:start");
        Bench bc=benchStatic(b,base,workDir,MNN_FORWARD_CPU,0,steps);
        markStage("run:cpu_static_train:done");
        req(bc.available&&bc.finite&&bc.checkpointReloadOk,
            "CPU static training/checkpoint reference failed");

        markStage("run:opencl_static_train:start");
        Bench bg=safeBenchStatic(
            b,base,workDir,MNN_FORWARD_OPENCL,
            MNN_GPU_TUNING_NORMAL|MNN_GPU_MEMORY_IMAGE,steps);
        markStage("run:opencl_static_train:done");

        Bench bv;
        bv.backend="VULKAN";
        bv.error="disabled_by_project_policy";

        const double cpuT=bc.tokps;
        const double clRatio=(cpuT>0&&bg.tokps>0)?bg.tokps/cpuT:0.0;
        const double vkRatio=0.0;

        // A training backend is accepted only after static correctness has
        // passed. Speed alone can never promote it.
        const bool clTrainOk=
            bg.available&&bg.finite&&bg.checkpointReloadOk&&bg.gpuOps>0;
        const bool clUseful=clTrainOk&&clRatio>=1.5;
        const bool clCanonical=clTrainOk&&clRatio>=2.0;
        const bool vkUseful=false;
        const bool vkCanonical=false;

        std::string winner="CPU";
        if(clCanonical){winner="OPENCL";}

        std::ostringstream o;
        o<<"{\"status\":\"PASS\",\"schema\":\"model0001_gpu_gate_report_v2\""
         <<",\"mnn_commit\":\""<<ANDROID_TRAINER_MNN_COMMIT<<"\""
         <<",\"checkpoint_sha256\":\""<<b.checkpointSha256<<"\""
         <<",\"model_state_sha256\":\""<<b.modelStateSha256<<"\""
         <<",\"rope_evidence\":\""<<jsonEscape(ropeEvidence)<<"\""
         <<",\"thermal_headroom_start\":"<<thermalHeadroom
         <<",\"opencl_runtime\":"<<openClProbe()
         <<",\"dynamic_parity\":{\"cpu\":"<<parityJson(cpu)
         <<",\"opencl\":"<<parityJson(clDynamic)<<"}"
         <<",\"static_parity\":{\"cpu\":"<<staticParityJson(cpuStatic)
         <<",\"opencl\":"<<staticParityJson(clStatic)<<"}"
         <<",\"static_train\":{\"cpu\":"<<benchJson(bc)
         <<",\"opencl\":"<<benchJson(bg)
         <<",\"vulkan_buffer\":"<<benchJson(bv)<<"}"
         <<",\"speed_ratio\":{\"opencl_vs_cpu\":"<<clRatio
         <<",\"vulkan_vs_cpu\":"<<vkRatio<<"}"
         <<",\"useful_1_5x\":{\"opencl\":"<<(clUseful?"true":"false")
         <<",\"vulkan_buffer\":"<<(vkUseful?"true":"false")<<"}"
         <<",\"canonical_2x\":{\"opencl\":"<<(clCanonical?"true":"false")
         <<",\"vulkan_buffer\":"<<(vkCanonical?"true":"false")<<"}"
         <<",\"recommended_backend\":\""<<winner<<"\""
         <<",\"note\":\"GPU correctness is validated through a memory-planned static FP32 graph exposing only locked scalar probes; dynamic OpenCL full-gradient host materialization is intentionally excluded after an Android LOW_MEMORY kill. Backend switch remains stage-boundary only.\"}";
        markStage("run:success");
        return o.str();
    } catch(const std::exception& e) {
        return std::string("{\"status\":\"FAIL\",\"error\":\"")+jsonEscape(e.what())+"\"}";
    }
}

} // namespace at
