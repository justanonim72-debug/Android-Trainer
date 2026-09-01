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
    // Interleaved [-x1,x0,-x3,x2,...] implemented by moving last dim to axis 0,
    // gathering paired indices, moving it back, then applying a fixed sign mask.
    std::vector<int32_t> idx(HD);
    std::vector<float> sign(HD);
    for (int i = 0; i < HD; i += 2) {
        idx[i] = i + 1; idx[i + 1] = i;
        sign[i] = -1.0f; sign[i + 1] = 1.0f;
    }
    auto xt = _Transpose(x, {3, 0, 1, 2});
    auto iv = _Const(idx.data(), {HD}, NHWC, halide_type_of<int32_t>());
    auto g = _GatherV2(xt, iv, _Scalar<int32_t>(0));
    auto back = _Transpose(g, {1, 2, 3, 0});
    auto sv = _Const(sign.data(), {1, 1, 1, HD}, NHWC);
    return back * sv;
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
    // [1,2,S,64] -> [1,2,1,S,64] -> tile repetition axis -> [1,6,S,64].
    auto e = _Reshape(x, {1,HKV,1,S,HD});
    auto reps = _Const(std::vector<int32_t>{1,1,HQ/HKV,1,1}.data(), {5}, NHWC, halide_type_of<int32_t>());
    return _Reshape(_Tile(e, reps), {1,HQ,S,HD});
}

VARP causalMask() {
    std::vector<float> m(S*S);
    for (int i=0;i<S;++i) for(int j=0;j<S;++j) m[i*S+j]=(j<=i)?0.0f:-1.0e9f;
    return _Const(m.data(), {1,1,S,S}, NHWC);
}

struct Graph {
    VARP tokenInput;
    VARP targetInput;
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
        auto ffv=_Silu(linear(h,gw))*linear(h,uw);
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
    BackendCounts counts;
    bool pass=false;
};

std::map<VARP,VARP> gradients(Graph& g) {
    OpGrad::init();
    std::set<VARP> ps;
    for(auto& kv:g.params) ps.insert(kv.second);
    auto m=OpGrad::grad(g.loss,ps);
    req(m.size()==g.params.size(),"MNN autograd did not return every parameter gradient");
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
    const float wd=static_cast<float>(b.adam.weightDecay);
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
        auto np=kv.second*scalar(1.0f-lr*wd)-scalar(lr)*adam;
        u.pnew.emplace(kv.first,np);
    }
    return u;
}

Parity dynamicParity(const Bundle& b,MNNForwardType type,int gpuMode) {
    Parity p; p.backend=typeName(type);
    auto exe=makeExecutor(type,4,gpuMode);
    req(exe!=nullptr,"cannot create "+p.backend+" executor");
    ExecutorScope scope(exe);
    installProfiler(exe,&p.counts);
    auto g=buildGraph(b); feedGraph(g,b);
    auto gm=gradients(g);
    auto upd=oneStepExpressions(g,gm,b);

    const float* lp=g.loss->readMap<float>(); req(lp,"loss readMap failed");
    p.loss=lp[0]; p.lossAbs=std::abs(p.loss-b.reference.loss);
    p.gradNorm=hostGlobalNorm(gm);
    p.gradNormRel=relerr(p.gradNorm,b.reference.globalGradNorm);

    const float* logits=g.logits->readMap<float>(); req(logits,"logits readMap failed");
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
            p.maxGradProbeAbs=std::max(p.maxGradProbeAbs,std::abs(got-ref));
            p.maxGradProbeRel=std::max(p.maxGradProbeRel,relerr(got,ref));
        }

        const float* nd=upd.pnew.at(slot)->readMap<float>(); req(nd,"AdamW update map failed");
        const auto& ra=refAdam(b,slot);
        const auto& ai=ra["probe_indices"]; const auto& av=ra["after"];
        for(rapidjson::SizeType i=0;i<ai.Size();++i) {
            int idx=ai[i].GetInt(); double ref=av[i].GetDouble(), got=nd[idx];
            p.maxAdamAbs=std::max(p.maxAdamAbs,std::abs(got-ref));
            p.maxAdamRel=std::max(p.maxAdamRel,relerr(got,ref));
        }
    }

    // CPU parity is intentionally strict; GPU still must remain close enough to
    // the PyTorch FP32 reference to be scientifically interchangeable at a stage boundary.
    p.pass=std::isfinite(p.loss)&&std::isfinite(p.gradNorm)
        && p.lossAbs<=2e-3 && p.maxLogitAbs<=5e-3
        && p.gradNormRel<=2e-2 && p.maxGradProbeAbs<=5e-3
        && p.maxAdamAbs<=5e-4;
    return p;
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
     <<",\"backend_counts\":{\"cpu\":"<<p.counts.cpu<<",\"opencl\":"<<p.counts.opencl
     <<",\"vulkan\":"<<p.counts.vulkan<<",\"other\":"<<p.counts.other
     <<",\"callbacks\":"<<p.counts.callbacks<<"}}";
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
    OpGrad::init();
    std::set<VARP> pset; for(auto& kv:g.params) pset.insert(kv.second);
    auto gm=OpGrad::grad(g.loss,pset);
    req(gm.size()==g.params.size(),"static AdamW: incomplete gradient map");

    VARP sum=scalar(0.0f);
    for(auto& kv:gm) sum=sum+_ReduceSum(_Square(kv.second),{},false);
    auto norm=_Sqrt(sum); norm->setName("global_grad_norm");
    auto coef=_Minimum(scalar(1.0f),scalar(1.0f)/(norm+scalar(1e-6f)));

    const float lr=static_cast<float>(b.adam.gateLr);
    const float b1=static_cast<float>(b.adam.beta1), b2=static_cast<float>(b.adam.beta2);
    const float eps=static_cast<float>(b.adam.eps), wd=static_cast<float>(b.adam.weightDecay);

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
    ParameterOptimizer::makeLoopModel(path.c_str(),{g.loss,norm},{oldState,newState});
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
};

Bench benchStatic(const Bundle& b,const std::string& baseModel,const std::string& workDir,
                  MNNForwardType type,int gpuMode,int steps) {
    Bench out; out.backend=typeName(type);
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(baseModel.c_str()),Interpreter::destroy);
    if(!net) return out;
    BackendConfig bc; bc.precision=BackendConfig::Precision_High; bc.power=BackendConfig::Power_High;
    ScheduleConfig cfg; cfg.type=type; cfg.numThread=(type==MNN_FORWARD_CPU)?4:1; cfg.backendConfig=&bc;
    cfg.mode=(type==MNN_FORWARD_OPENCL)?gpuMode:0;
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
    net->releaseSession(session);
    return out;
}

std::string benchJson(const Bench& b) {
    std::ostringstream o;
    o<<"{\"backend\":\""<<b.backend<<"\",\"available\":"<<(b.available?"true":"false")
      <<",\"finite\":"<<(b.finite?"true":"false")<<",\"steps\":"<<b.steps
      <<",\"first_loss\":"<<b.firstLoss<<",\"last_loss\":"<<b.lastLoss
      <<",\"seconds\":"<<b.seconds<<",\"target_tokens_per_second\":"<<b.tokps
      <<",\"profile\":{\"cpu_backend_hits\":"<<b.cpuOps<<",\"gpu_backend_hits\":"<<b.gpuOps
      <<",\"other_backend_hits\":"<<b.otherOps<<"},\"checkpoint\":\""<<jsonEscape(b.checkpointPath)<<"\"}";
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
        return o.str();
    } catch(const std::exception& e) {
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
    try {
        auto b=Bundle::load(dir);

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
            cpu = dynamicParity(b,MNN_FORWARD_CPU,0);
        }
        if(!cpu.pass) {
            return std::string("{\"status\":\"FAIL_CPU_PARITY\",\"thermal_headroom_start\":")+
                std::to_string(thermalHeadroom)+",\"cpu_parity\":"+parityJson(cpu)+"}";
        }

        // 2) Exact dynamic graph parity on GPU candidates.
        Parity cl=dynamicParity(b,MNN_FORWARD_OPENCL,MNN_GPU_TUNING_FAST|MNN_GPU_MEMORY_IMAGE);
        Parity vk=dynamicParity(b,MNN_FORWARD_VULKAN,MNN_GPU_TUNING_WIDE);

        // 3) Serialize one backend-neutral static AdamW training loop and benchmark
        // independent fresh sessions. This avoids timing MNN's dynamic graph rebuild path.
        const std::string base=workDir+"/model0001-gate-static-base.mnn";
        buildStaticAdamWModel(b,base);
        const int steps=20;
        Bench bc=benchStatic(b,base,workDir,MNN_FORWARD_CPU,0,steps);
        Bench bg;
        if(cl.pass) bg=benchStatic(b,base,workDir,MNN_FORWARD_OPENCL,
                                   MNN_GPU_TUNING_NORMAL|MNN_GPU_MEMORY_IMAGE,steps);
        else { bg.backend="OPENCL"; }
        Bench bv;
        if(vk.pass) bv=benchStatic(b,base,workDir,MNN_FORWARD_VULKAN,MNN_GPU_TUNING_WIDE,steps);
        else { bv.backend="VULKAN"; }

        const double cpuT=bc.tokps;
        const double clRatio=(cpuT>0&&bg.tokps>0)?bg.tokps/cpuT:0.0;
        const double vkRatio=(cpuT>0&&bv.tokps>0)?bv.tokps/cpuT:0.0;

        // Acceptance is deliberately evidence-based, not "GPU did not crash".
        // 1.5x is reported as useful; 2x is the recommended canonical-switch threshold.
        const bool clUseful=cl.pass&&bg.finite&&clRatio>=1.5;
        const bool clCanonical=cl.pass&&bg.finite&&clRatio>=2.0;
        const bool vkUseful=vk.pass&&bv.finite&&vkRatio>=1.5;
        const bool vkCanonical=vk.pass&&bv.finite&&vkRatio>=2.0;

        std::string winner="CPU";
        double best=1.0;
        if(clCanonical&&clRatio>best){winner="OPENCL";best=clRatio;}
        if(vkCanonical&&vkRatio>best){winner="VULKAN_BUFFER";best=vkRatio;}

        std::ostringstream o;
        o<<"{\"status\":\"PASS\",\"schema\":\"model0001_gpu_gate_report_v1\""
         <<",\"mnn_commit\":\""<<ANDROID_TRAINER_MNN_COMMIT<<"\""
         <<",\"checkpoint_sha256\":\""<<b.checkpointSha256<<"\""
         <<",\"rope_evidence\":\""<<jsonEscape(ropeEvidence)<<"\""
         <<",\"thermal_headroom_start\":"<<thermalHeadroom
         <<",\"opencl_runtime\":"<<openClProbe()
         <<",\"parity\":{\"cpu\":"<<parityJson(cpu)<<",\"opencl\":"<<parityJson(cl)
         <<",\"vulkan_buffer\":"<<parityJson(vk)<<"}"
         <<",\"static_train\":{\"cpu\":"<<benchJson(bc)<<",\"opencl\":"<<benchJson(bg)
         <<",\"vulkan_buffer\":"<<benchJson(bv)<<"}"
         <<",\"speed_ratio\":{\"opencl_vs_cpu\":"<<clRatio<<",\"vulkan_vs_cpu\":"<<vkRatio<<"}"
         <<",\"useful_1_5x\":{\"opencl\":"<<(clUseful?"true":"false")
         <<",\"vulkan_buffer\":"<<(vkUseful?"true":"false")<<"}"
         <<",\"canonical_2x\":{\"opencl\":"<<(clCanonical?"true":"false")
         <<",\"vulkan_buffer\":"<<(vkCanonical?"true":"false")<<"}"
         <<",\"recommended_backend\":\""<<winner<<"\""
         <<",\"note\":\"Benchmark is exact Model #0001 FP32 forward/backward/decoupled-AdamW; backend switch remains stage-boundary only.\"}";
        return o.str();
    } catch(const std::exception& e) {
        return std::string("{\"status\":\"FAIL\",\"error\":\"")+jsonEscape(e.what())+"\"}";
    }
}

} // namespace at
