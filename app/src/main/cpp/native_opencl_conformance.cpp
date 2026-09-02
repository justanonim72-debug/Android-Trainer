#include "native_opencl_conformance.hpp"

#include <CL/cl.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace at {
namespace {

void req(bool v,const std::string& m){if(!v)throw std::runtime_error(m);}

struct Api {
    void* lib=nullptr;
    decltype(&clGetPlatformIDs) GetPlatformIDs=nullptr;
    decltype(&clGetDeviceIDs) GetDeviceIDs=nullptr;
    decltype(&clGetDeviceInfo) GetDeviceInfo=nullptr;
    decltype(&clCreateContext) CreateContext=nullptr;
    decltype(&clCreateCommandQueue) CreateCommandQueue=nullptr;
    decltype(&clCreateProgramWithSource) CreateProgramWithSource=nullptr;
    decltype(&clBuildProgram) BuildProgram=nullptr;
    decltype(&clGetProgramBuildInfo) GetProgramBuildInfo=nullptr;
    decltype(&clCreateKernel) CreateKernel=nullptr;
    decltype(&clCreateBuffer) CreateBuffer=nullptr;
    decltype(&clSetKernelArg) SetKernelArg=nullptr;
    decltype(&clEnqueueWriteBuffer) EnqueueWriteBuffer=nullptr;
    decltype(&clEnqueueReadBuffer) EnqueueReadBuffer=nullptr;
    decltype(&clEnqueueNDRangeKernel) EnqueueNDRangeKernel=nullptr;
    decltype(&clFinish) Finish=nullptr;

    template<class T> void sym(T& p,const char* n){
        p=reinterpret_cast<T>(dlsym(lib,n));
        req(p!=nullptr,std::string("missing OpenCL symbol ")+n);
    }
    void load(){
        if(lib)return;
        const char* libs[]={"libOpenCL.so","libGLES_mali.so","libmali.so"};
        for(auto* n:libs){lib=dlopen(n,RTLD_NOW|RTLD_LOCAL);if(lib)break;}
        req(lib!=nullptr,"cannot dlopen Android OpenCL library");
        sym(GetPlatformIDs,"clGetPlatformIDs");
        sym(GetDeviceIDs,"clGetDeviceIDs");
        sym(GetDeviceInfo,"clGetDeviceInfo");
        sym(CreateContext,"clCreateContext");
        sym(CreateCommandQueue,"clCreateCommandQueue");
        sym(CreateProgramWithSource,"clCreateProgramWithSource");
        sym(BuildProgram,"clBuildProgram");
        sym(GetProgramBuildInfo,"clGetProgramBuildInfo");
        sym(CreateKernel,"clCreateKernel");
        sym(CreateBuffer,"clCreateBuffer");
        sym(SetKernelArg,"clSetKernelArg");
        sym(EnqueueWriteBuffer,"clEnqueueWriteBuffer");
        sym(EnqueueReadBuffer,"clEnqueueReadBuffer");
        sym(EnqueueNDRangeKernel,"clEnqueueNDRangeKernel");
        sym(Finish,"clFinish");
    }
};

static const char* kSource=R"CLC(
#pragma OPENCL FP_CONTRACT OFF

__kernel void vec_add(__global const float* a,__global const float* b,__global float* y,int n){
    int i=(int)get_global_id(0); if(i<n)y[i]=a[i]+b[i];
}

__kernel void silu_fb(__global const float* x,__global const float* dy,
                      __global float* y,__global float* dx,int n){
    int i=(int)get_global_id(0); if(i>=n)return;
    float s=1.0f/(1.0f+exp(-x[i]));
    y[i]=x[i]*s;
    dx[i]=dy[i]*(s+x[i]*s*(1.0f-s));
}

__kernel void rope_interleaved_fb(__global const float* x,__global const float* dy,
                                  __global const float* cs,__global const float* sn,
                                  __global float* y,__global float* dx,
                                  int rows,int hd){
    int g=(int)get_global_id(0);
    int pairs=rows*(hd/2); if(g>=pairs)return;
    int row=g/(hd/2), pair=g%(hd/2);
    int i0=row*hd+2*pair, i1=i0+1;
    float c=cs[i0],s=sn[i0];
    float a=x[i0],b=x[i1],u=dy[i0],v=dy[i1];
    y[i0]=a*c-b*s; y[i1]=b*c+a*s;
    dx[i0]=u*c+v*s; dx[i1]=-u*s+v*c;
}

__kernel void rmsnorm_fb(__global const float* x,__global const float* w,
                         __global const float* dy,__global float* y,
                         __global float* dx,int rows,int d,float eps){
    int r=(int)get_global_id(0); if(r>=rows)return;
    int base=r*d;
    float ss=0.0f;
    for(int j=0;j<d;++j){float z=x[base+j];ss += z*z;}
    float inv=1.0f/sqrt(ss/(float)d+eps);
    float dot=0.0f;
    for(int j=0;j<d;++j)dot += dy[base+j]*w[j]*x[base+j];
    float corr=dot/(float)d*inv*inv*inv;
    for(int j=0;j<d;++j){
        float z=x[base+j];
        y[base+j]=z*inv*w[j];
        dx[base+j]=dy[base+j]*w[j]*inv-z*corr;
    }
}

__kernel void rmsnorm_dw(__global const float* x,__global const float* dy,
                         __global float* dw,int rows,int d,float eps){
    int j=(int)get_global_id(0); if(j>=d)return;
    float acc=0.0f;
    for(int r=0;r<rows;++r){
        int base=r*d;
        float ss=0.0f;
        for(int k=0;k<d;++k){float z=x[base+k];ss += z*z;}
        float inv=1.0f/sqrt(ss/(float)d+eps);
        acc += dy[base+j]*x[base+j]*inv;
    }
    dw[j]=acc;
}

__kernel void gemm_nt_probe(__global const float* a,__global const float* w,
                            __global const int* pm,__global const int* pn,
                            __global float* out,int probes,int k,int n){
    int p=(int)get_global_id(0); if(p>=probes)return;
    int m=pm[p],nn=pn[p]; float acc=0.0f;
    int ab=m*k,wb=nn*k;
    for(int kk=0;kk<k;++kk)acc += a[ab+kk]*w[wb+kk];
    out[p]=acc;
}

__kernel void gemm_dinput_probe(__global const float* dy,__global const float* w,
                                __global const int* pm,__global const int* pk,
                                __global float* out,int probes,int k,int n){
    int p=(int)get_global_id(0); if(p>=probes)return;
    int m=pm[p],kk=pk[p]; float acc=0.0f;
    int db=m*n;
    for(int nn=0;nn<n;++nn)acc += dy[db+nn]*w[nn*k+kk];
    out[p]=acc;
}

__kernel void gemm_dweight_probe(__global const float* dy,__global const float* a,
                                 __global const int* pn,__global const int* pk,
                                 __global float* out,int probes,int m,int k,int n){
    int p=(int)get_global_id(0); if(p>=probes)return;
    int nn=pn[p],kk=pk[p]; float acc=0.0f;
    for(int mm=0;mm<m;++mm)acc += dy[mm*n+nn]*a[mm*k+kk];
    out[p]=acc;
}

__kernel void causal_softmax_fb(__global const float* x,__global const float* dy,
                                __global float* y,__global float* dx,int rows,int s){
    int r=(int)get_global_id(0); if(r>=rows)return;
    int pos=r%s,base=r*s;
    float mx=-3.402823466e+38F;
    for(int j=0;j<=pos;++j)mx=fmax(mx,x[base+j]);
    float den=0.0f;
    for(int j=0;j<=pos;++j)den += exp(x[base+j]-mx);
    float dot=0.0f;
    for(int j=0;j<s;++j){
        float v=(j<=pos)?exp(x[base+j]-mx)/den:0.0f;
        y[base+j]=v; dot += dy[base+j]*v;
    }
    for(int j=0;j<s;++j){
        float v=y[base+j];
        dx[base+j]=(j<=pos)?v*(dy[base+j]-dot):0.0f;
    }
}

__kernel void gqa_reduce(__global const float* src,__global float* dst,
                         int hq,int hkv,int s,int hd){
    int g=(int)get_global_id(0);
    int total=hkv*s*hd;if(g>=total)return;
    int d=g%hd; int t=g/hd; int p=t%s; int kv=t/s;
    int rep=hq/hkv; float acc=0.0f;
    for(int r=0;r<rep;++r){
        int qh=kv*rep+r;
        acc += src[(qh*s+p)*hd+d];
    }
    dst[g]=acc;
}

__kernel void embedding_reduce(__global const float* lookup_grad,
                               __global const int* positions,
                               __global const int* offsets,
                               __global float* out,int unique_count,int d){
    int g=(int)get_global_id(0);
    int total=unique_count*d;if(g>=total)return;
    int u=g/d,j=g%d; float acc=0.0f;
    for(int q=offsets[u];q<offsets[u+1];++q)acc += lookup_grad[positions[q]*d+j];
    out[g]=acc;
}

__kernel void cross_entropy_rows(__global const float* logits,__global const int* target,
                                 __global float* row_loss,int rows,int v){
    int r=(int)get_global_id(0);if(r>=rows)return;
    int base=r*v; float mx=-3.402823466e+38F;
    for(int j=0;j<v;++j)mx=fmax(mx,logits[base+j]);
    float den=0.0f;
    for(int j=0;j<v;++j)den += exp(logits[base+j]-mx);
    row_loss[r]=-(logits[base+target[r]]-mx-log(den));
}

__kernel void gradnorm_serial(__global const float* g,__global float* norm,int n){
    if(get_global_id(0)!=0)return;
    float ss=0.0f;for(int i=0;i<n;++i)ss += g[i]*g[i];
    norm[0]=sqrt(ss);
}

__kernel void adamw_step(__global const float* p,__global const float* g,
                         __global const float* m,__global const float* v,
                         __global float* po,__global float* mo,__global float* vo,
                         int n,float lr,float b1,float b2,float eps,float wd,
                         float clip){
    int i=(int)get_global_id(0);if(i>=n)return;
    float gg=g[i]*clip;
    float mn=b1*m[i]+(1.0f-b1)*gg;
    float vn=b2*v[i]+(1.0f-b2)*gg*gg;
    float mhat=mn/(1.0f-b1);
    float vhat=vn/(1.0f-b2);
    po[i]=p[i]*(1.0f-lr*wd)-lr*mhat/(sqrt(vhat)+eps);
    mo[i]=mn;vo[i]=vn;
}
)CLC";

struct Runtime {
    Api api;
    cl_platform_id platform=nullptr;
    cl_device_id device=nullptr;
    cl_context context=nullptr;
    cl_command_queue queue=nullptr;
    cl_program program=nullptr;
    std::string deviceName;
    std::string openclC;
    bool ready=false;

    void init(){
        if(ready)return;
        api.load();
        cl_uint np=0;req(api.GetPlatformIDs(0,nullptr,&np)==CL_SUCCESS&&np>0,"no OpenCL platform");
        std::vector<cl_platform_id> ps(np);req(api.GetPlatformIDs(np,ps.data(),nullptr)==CL_SUCCESS,"platform query failed");
        for(auto p:ps){
            cl_uint nd=0;cl_device_id d=nullptr;
            if(api.GetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&d,&nd)==CL_SUCCESS&&nd>0){platform=p;device=d;break;}
        }
        req(device!=nullptr,"no OpenCL GPU device");
        auto getStr=[&](cl_device_info what){
            size_t n=0;api.GetDeviceInfo(device,what,0,nullptr,&n);
            std::string s(n? n:1,'\0');
            if(n)api.GetDeviceInfo(device,what,n,s.data(),nullptr);
            while(!s.empty()&&s.back()=='\0')s.pop_back();
            return s;
        };
        deviceName=getStr(CL_DEVICE_NAME);
        openclC=getStr(CL_DEVICE_OPENCL_C_VERSION);
        cl_int e=CL_SUCCESS;
        const cl_context_properties props[]={
            CL_CONTEXT_PLATFORM,reinterpret_cast<cl_context_properties>(platform),0
        };
        context=api.CreateContext(props,1,&device,nullptr,nullptr,&e);
        req(e==CL_SUCCESS&&context,"clCreateContext failed ec="+std::to_string(e));
        queue=api.CreateCommandQueue(context,device,0,&e);
        req(e==CL_SUCCESS&&queue,"clCreateCommandQueue failed ec="+std::to_string(e));
        const char* src=kSource;size_t len=std::strlen(src);
        program=api.CreateProgramWithSource(context,1,&src,&len,&e);
        req(e==CL_SUCCESS&&program,"clCreateProgramWithSource failed ec="+std::to_string(e));
        e=api.BuildProgram(program,1,&device,"-cl-std=CL1.2",nullptr,nullptr);
        if(e!=CL_SUCCESS){
            size_t n=0;api.GetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,nullptr,&n);
            std::string log(n? n:1,'\0');
            if(n)api.GetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,n,log.data(),nullptr);
            throw std::runtime_error("OpenCL build failed ec="+std::to_string(e)+" log="+log);
        }
        ready=true;
    }

    cl_mem buf(size_t bytes,cl_mem_flags flags=CL_MEM_READ_WRITE){
        cl_int e=CL_SUCCESS;auto b=api.CreateBuffer(context,flags,bytes,nullptr,&e);
        req(e==CL_SUCCESS&&b,"clCreateBuffer failed ec="+std::to_string(e));
        return b;
    }
    void write(cl_mem b,const void* p,size_t bytes){
        auto e=api.EnqueueWriteBuffer(queue,b,CL_TRUE,0,bytes,p,0,nullptr,nullptr);
        req(e==CL_SUCCESS,"clEnqueueWriteBuffer failed ec="+std::to_string(e));
    }
    void read(cl_mem b,void* p,size_t bytes){
        auto e=api.EnqueueReadBuffer(queue,b,CL_TRUE,0,bytes,p,0,nullptr,nullptr);
        req(e==CL_SUCCESS,"clEnqueueReadBuffer failed ec="+std::to_string(e));
    }
    cl_kernel kernel(const char* n){
        cl_int e=CL_SUCCESS;auto k=api.CreateKernel(program,n,&e);
        req(e==CL_SUCCESS&&k,std::string("clCreateKernel failed ")+n+" ec="+std::to_string(e));
        return k;
    }
    template<class T> void arg(cl_kernel k,cl_uint i,const T& v){
        auto e=api.SetKernelArg(k,i,sizeof(T),&v);
        req(e==CL_SUCCESS,"clSetKernelArg failed ec="+std::to_string(e));
    }
    void run1(cl_kernel k,size_t n){
        size_t g=n;auto e=api.EnqueueNDRangeKernel(queue,k,1,nullptr,&g,nullptr,0,nullptr,nullptr);
        req(e==CL_SUCCESS,"clEnqueueNDRangeKernel failed ec="+std::to_string(e));
        e=api.Finish(queue);req(e==CL_SUCCESS,"clFinish failed ec="+std::to_string(e));
    }
};

static Runtime* gRuntime=nullptr;
static std::string gCached;

Runtime& rt(){
    if(!gRuntime)gRuntime=new Runtime(); // process lifetime by design
    gRuntime->init();return *gRuntime;
}

struct Rng{
    uint32_t s=0x4d595df4u;
    uint32_t next(){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
    float f(float scale=0.5f){return ((int32_t)(next()&0xffff)-32768)*(scale/32768.0f);}
};

struct Test{
    std::string name;
    double maxAbs=0,maxRel=0,absTol=0,relTol=0;
    bool pass=false;
};

Test cmp(const std::string& name,const std::vector<float>& got,const std::vector<float>& ref,double at,double rt0){
    req(got.size()==ref.size(),"compare size mismatch "+name);
    Test t{name,0,0,at,rt0,false};
    for(size_t i=0;i<got.size();++i){
        if(!std::isfinite(got[i])||!std::isfinite(ref[i])){
            t.maxAbs=std::numeric_limits<double>::infinity();
            t.maxRel=std::numeric_limits<double>::infinity();break;
        }
        double a=std::abs((double)got[i]-ref[i]);
        double d=std::max(1e-6,std::abs((double)ref[i]));
        t.maxAbs=std::max(t.maxAbs,a);t.maxRel=std::max(t.maxRel,a/d);
    }
    t.pass=t.maxAbs<=at||t.maxRel<=rt0;
    return t;
}

std::string esc(const std::string& s){
    std::string o;for(char c:s){if(c=='"'||c=='\\')o.push_back('\\');o.push_back(c);}return o;
}

void add(std::vector<Test>& ts,const Test& t){ts.push_back(t);}

void testVec(Runtime& r,std::vector<Test>& ts,Rng& q){
    int n=4096;std::vector<float>a(n),b(n),ref(n),got(n);
    for(int i=0;i<n;++i){a[i]=q.f();b[i]=q.f();ref[i]=a[i]+b[i];}
    auto ba=r.buf(n*4),bb=r.buf(n*4),bo=r.buf(n*4);r.write(ba,a.data(),n*4);r.write(bb,b.data(),n*4);
    auto k=r.kernel("vec_add");r.arg(k,0,ba);r.arg(k,1,bb);r.arg(k,2,bo);r.arg(k,3,n);r.run1(k,n);r.read(bo,got.data(),n*4);
    add(ts,cmp("vector_add_fp32",got,ref,1e-7,1e-6));
}

void testSilu(Runtime& r,std::vector<Test>& ts,Rng& q){
    int n=4096;std::vector<float>x(n),dy(n),y(n),dx(n),ry(n),rdx(n);
    for(int i=0;i<n;++i){x[i]=q.f(3);dy[i]=q.f();float s=1.f/(1.f+std::exp(-x[i]));ry[i]=x[i]*s;rdx[i]=dy[i]*(s+x[i]*s*(1-s));}
    auto bx=r.buf(n*4),bd=r.buf(n*4),by=r.buf(n*4),bdo=r.buf(n*4);r.write(bx,x.data(),n*4);r.write(bd,dy.data(),n*4);
    auto k=r.kernel("silu_fb");r.arg(k,0,bx);r.arg(k,1,bd);r.arg(k,2,by);r.arg(k,3,bdo);r.arg(k,4,n);r.run1(k,n);
    r.read(by,y.data(),n*4);r.read(bdo,dx.data(),n*4);
    add(ts,cmp("silu_forward",y,ry,2e-6,2e-5));add(ts,cmp("silu_backward",dx,rdx,3e-6,3e-5));
}

void testRope(Runtime& r,std::vector<Test>& ts,Rng& q){
    const int S=256,H=6,HD=64,rows=S*H,n=rows*HD;
    std::vector<float>x(n),dy(n),cs(n),sn(n),y(n),dx(n),ry(n),rdx(n);
    for(int row=0;row<rows;++row)for(int p=0;p<HD/2;++p){
        int i0=row*HD+2*p,i1=i0+1;int pos=row%S;
        float ang=(float)(pos*std::pow(10000.0,-2.0*p/HD));
        float c=std::cos(ang),s=std::sin(ang);cs[i0]=cs[i1]=c;sn[i0]=sn[i1]=s;
        x[i0]=q.f();x[i1]=q.f();dy[i0]=q.f();dy[i1]=q.f();
        ry[i0]=x[i0]*c-x[i1]*s;ry[i1]=x[i1]*c+x[i0]*s;
        rdx[i0]=dy[i0]*c+dy[i1]*s;rdx[i1]=-dy[i0]*s+dy[i1]*c;
    }
    auto bx=r.buf(n*4),bd=r.buf(n*4),bc=r.buf(n*4),bs=r.buf(n*4),by=r.buf(n*4),bdo=r.buf(n*4);
    r.write(bx,x.data(),n*4);r.write(bd,dy.data(),n*4);r.write(bc,cs.data(),n*4);r.write(bs,sn.data(),n*4);
    auto k=r.kernel("rope_interleaved_fb");r.arg(k,0,bx);r.arg(k,1,bd);r.arg(k,2,bc);r.arg(k,3,bs);r.arg(k,4,by);r.arg(k,5,bdo);r.arg(k,6,rows);r.arg(k,7,HD);r.run1(k,rows*(HD/2));
    r.read(by,y.data(),n*4);r.read(bdo,dx.data(),n*4);
    add(ts,cmp("rope_interleaved_forward",y,ry,3e-6,3e-5));add(ts,cmp("rope_interleaved_backward",dx,rdx,3e-6,3e-5));
}

void testRms(Runtime& r,std::vector<Test>& ts,Rng& q){
    const int rows=256,d=384,n=rows*d;const float eps=1e-5f;
    std::vector<float>x(n),dy(n),w(d),y(n),dx(n),dw(d),ry(n),rdx(n),rdw(d,0);
    for(auto&v:x)v=q.f();for(auto&v:dy)v=q.f();for(auto&v:w)v=0.8f+q.f(0.2f);
    for(int rr=0;rr<rows;++rr){
        int b=rr*d;float ss=0;for(int j=0;j<d;++j)ss+=x[b+j]*x[b+j];
        float inv=1.f/std::sqrt(ss/d+eps);float dot=0;for(int j=0;j<d;++j)dot+=dy[b+j]*w[j]*x[b+j];
        float corr=dot/d*inv*inv*inv;
        for(int j=0;j<d;++j){ry[b+j]=x[b+j]*inv*w[j];rdx[b+j]=dy[b+j]*w[j]*inv-x[b+j]*corr;rdw[j]+=dy[b+j]*x[b+j]*inv;}
    }
    auto bx=r.buf(n*4),bw=r.buf(d*4),bd=r.buf(n*4),by=r.buf(n*4),bdo=r.buf(n*4),bdw=r.buf(d*4);
    r.write(bx,x.data(),n*4);r.write(bw,w.data(),d*4);r.write(bd,dy.data(),n*4);
    auto k=r.kernel("rmsnorm_fb");r.arg(k,0,bx);r.arg(k,1,bw);r.arg(k,2,bd);r.arg(k,3,by);r.arg(k,4,bdo);r.arg(k,5,rows);r.arg(k,6,d);r.arg(k,7,eps);r.run1(k,rows);
    auto kd=r.kernel("rmsnorm_dw");r.arg(kd,0,bx);r.arg(kd,1,bd);r.arg(kd,2,bdw);r.arg(kd,3,rows);r.arg(kd,4,d);r.arg(kd,5,eps);r.run1(kd,d);
    r.read(by,y.data(),n*4);r.read(bdo,dx.data(),n*4);r.read(bdw,dw.data(),d*4);
    add(ts,cmp("rmsnorm_forward",y,ry,5e-5,2e-4));add(ts,cmp("rmsnorm_dx",dx,rdx,8e-5,3e-4));add(ts,cmp("rmsnorm_dw",dw,rdw,2e-4,5e-4));
}

void testGemmShape(Runtime& r,std::vector<Test>& ts,Rng& q,const std::string& tag,int M,int K,int N,bool doDInput){
    const int P=24;
    std::vector<float>a((size_t)M*K),w((size_t)N*K),dy((size_t)M*N);
    for(auto&v:a)v=q.f(0.15f);for(auto&v:w)v=q.f(0.15f);for(auto&v:dy)v=q.f(0.1f);
    std::vector<int>pm(P),pn(P),pk(P);for(int i=0;i<P;++i){pm[i]=(i*37+5)%M;pn[i]=(i*97+11)%N;pk[i]=(i*53+7)%K;}
    auto ba=r.buf(a.size()*4),bw=r.buf(w.size()*4),bdy=r.buf(dy.size()*4),bpm=r.buf(P*4),bpn=r.buf(P*4),bpk=r.buf(P*4),bo=r.buf(P*4);
    r.write(ba,a.data(),a.size()*4);r.write(bw,w.data(),w.size()*4);r.write(bdy,dy.data(),dy.size()*4);r.write(bpm,pm.data(),P*4);r.write(bpn,pn.data(),P*4);r.write(bpk,pk.data(),P*4);
    std::vector<float>got(P),ref(P);
    for(int p=0;p<P;++p){float s=0;for(int kk=0;kk<K;++kk)s+=a[(size_t)pm[p]*K+kk]*w[(size_t)pn[p]*K+kk];ref[p]=s;}
    auto k=r.kernel("gemm_nt_probe");r.arg(k,0,ba);r.arg(k,1,bw);r.arg(k,2,bpm);r.arg(k,3,bpn);r.arg(k,4,bo);r.arg(k,5,P);r.arg(k,6,K);r.arg(k,7,N);r.run1(k,P);r.read(bo,got.data(),P*4);
    add(ts,cmp("gemm_fwd_"+tag,got,ref,3e-4,5e-4));
    if(doDInput){
        for(int p=0;p<P;++p){float s=0;for(int nn=0;nn<N;++nn)s+=dy[(size_t)pm[p]*N+nn]*w[(size_t)nn*K+pk[p]];ref[p]=s;}
        auto kd=r.kernel("gemm_dinput_probe");r.arg(kd,0,bdy);r.arg(kd,1,bw);r.arg(kd,2,bpm);r.arg(kd,3,bpk);r.arg(kd,4,bo);r.arg(kd,5,P);r.arg(kd,6,K);r.arg(kd,7,N);r.run1(kd,P);r.read(bo,got.data(),P*4);
        add(ts,cmp("gemm_dinput_"+tag,got,ref,5e-4,8e-4));
    }
    for(int p=0;p<P;++p){float s=0;for(int mm=0;mm<M;++mm)s+=dy[(size_t)mm*N+pn[p]]*a[(size_t)mm*K+pk[p]];ref[p]=s;}
    auto kw=r.kernel("gemm_dweight_probe");r.arg(kw,0,bdy);r.arg(kw,1,ba);r.arg(kw,2,bpn);r.arg(kw,3,bpk);r.arg(kw,4,bo);r.arg(kw,5,P);r.arg(kw,6,M);r.arg(kw,7,K);r.arg(kw,8,N);r.run1(kw,P);r.read(bo,got.data(),P*4);
    add(ts,cmp("gemm_dweight_"+tag,got,ref,4e-4,8e-4));
}

void testSoftmax(Runtime& r,std::vector<Test>& ts,Rng& q){
    const int S=256,H=6,rows=S*H,n=rows*S;
    std::vector<float>x(n),dy(n),y(n),dx(n),ry(n),rdx(n);
    for(auto&v:x)v=q.f(2);for(auto&v:dy)v=q.f();
    for(int rr=0;rr<rows;++rr){
        int pos=rr%S,b=rr*S;float mx=-1e30f;for(int j=0;j<=pos;++j)mx=std::max(mx,x[b+j]);
        float den=0;for(int j=0;j<=pos;++j)den+=std::exp(x[b+j]-mx);
        float dot=0;for(int j=0;j<S;++j){ry[b+j]=(j<=pos)?std::exp(x[b+j]-mx)/den:0;dot+=dy[b+j]*ry[b+j];}
        for(int j=0;j<S;++j)rdx[b+j]=(j<=pos)?ry[b+j]*(dy[b+j]-dot):0;
    }
    auto bx=r.buf(n*4),bd=r.buf(n*4),by=r.buf(n*4),bdo=r.buf(n*4);r.write(bx,x.data(),n*4);r.write(bd,dy.data(),n*4);
    auto k=r.kernel("causal_softmax_fb");r.arg(k,0,bx);r.arg(k,1,bd);r.arg(k,2,by);r.arg(k,3,bdo);r.arg(k,4,rows);r.arg(k,5,S);r.run1(k,rows);
    r.read(by,y.data(),n*4);r.read(bdo,dx.data(),n*4);
    add(ts,cmp("causal_softmax_forward",y,ry,2e-5,2e-4));add(ts,cmp("causal_softmax_backward",dx,rdx,2e-5,3e-4));
}

void testGqa(Runtime& r,std::vector<Test>& ts,Rng& q){
    const int HQ=6,HKV=2,S=256,HD=64,n=HQ*S*HD,o=HKV*S*HD;
    std::vector<float>x(n),got(o),ref(o,0);for(auto&v:x)v=q.f();
    for(int kv=0;kv<HKV;++kv)for(int p=0;p<S;++p)for(int d=0;d<HD;++d)
        for(int rr=0;rr<HQ/HKV;++rr)ref[(kv*S+p)*HD+d]+=x[((kv*(HQ/HKV)+rr)*S+p)*HD+d];
    auto bx=r.buf(n*4),bo=r.buf(o*4);r.write(bx,x.data(),n*4);
    auto k=r.kernel("gqa_reduce");r.arg(k,0,bx);r.arg(k,1,bo);r.arg(k,2,HQ);r.arg(k,3,HKV);r.arg(k,4,S);r.arg(k,5,HD);r.run1(k,o);r.read(bo,got.data(),o*4);
    add(ts,cmp("gqa_backward_reduce",got,ref,1e-7,1e-6));
}

void testEmbedding(Runtime& r,std::vector<Test>& ts,Rng& q){
    const int S=256,D=384,U=31;
    std::vector<float>lg(S*D);for(auto&v:lg)v=q.f();
    std::vector<std::vector<int>>lists(U);for(int p=0;p<S;++p)lists[(p*17+3)%U].push_back(p);
    std::vector<int>off(U+1),pos;for(int u=0;u<U;++u){off[u]=pos.size();pos.insert(pos.end(),lists[u].begin(),lists[u].end());}off[U]=pos.size();
    std::vector<float>ref(U*D,0),got(U*D);
    for(int u=0;u<U;++u)for(int d=0;d<D;++d)for(int z:lists[u])ref[u*D+d]+=lg[z*D+d];
    auto bl=r.buf(lg.size()*4),bp=r.buf(pos.size()*4),boff=r.buf(off.size()*4),bout=r.buf(got.size()*4);
    r.write(bl,lg.data(),lg.size()*4);r.write(bp,pos.data(),pos.size()*4);r.write(boff,off.data(),off.size()*4);
    auto k=r.kernel("embedding_reduce");r.arg(k,0,bl);r.arg(k,1,bp);r.arg(k,2,boff);r.arg(k,3,bout);r.arg(k,4,U);r.arg(k,5,D);r.run1(k,got.size());r.read(bout,got.data(),got.size()*4);
    add(ts,cmp("embedding_gradient_reduce_no_atomics",got,ref,2e-6,2e-5));
}

void testCe(Runtime& r,std::vector<Test>& ts,Rng& q){
    const int rows=8,V=14000;std::vector<float>l((size_t)rows*V);std::vector<int>t(rows);std::vector<float>ref(rows),got(rows);
    for(auto&v:l)v=q.f(3);for(int rr=0;rr<rows;++rr){t[rr]=(rr*1777+13)%V;float mx=-1e30f;for(int j=0;j<V;++j)mx=std::max(mx,l[(size_t)rr*V+j]);float den=0;for(int j=0;j<V;++j)den+=std::exp(l[(size_t)rr*V+j]-mx);ref[rr]=-(l[(size_t)rr*V+t[rr]]-mx-std::log(den));}
    auto bl=r.buf(l.size()*4),bt=r.buf(t.size()*4),bo=r.buf(rows*4);r.write(bl,l.data(),l.size()*4);r.write(bt,t.data(),t.size()*4);
    auto k=r.kernel("cross_entropy_rows");r.arg(k,0,bl);r.arg(k,1,bt);r.arg(k,2,bo);r.arg(k,3,rows);r.arg(k,4,V);r.run1(k,rows);r.read(bo,got.data(),rows*4);
    add(ts,cmp("cross_entropy_vocab14000",got,ref,3e-4,5e-5));
}

void testOpt(Runtime& r,std::vector<Test>& ts,Rng& q){
    const int n=4096;std::vector<float>p(n),g(n),m(n,0),v(n,0),po(n),mo(n),vo(n),rpo(n),rmo(n),rvo(n),norm(1),rnorm(1);
    for(int i=0;i<n;++i){p[i]=q.f();g[i]=q.f();}
    float ss=0;for(float z:g)ss+=z*z;rnorm[0]=std::sqrt(ss);
    auto bp=r.buf(n*4),bg=r.buf(n*4),bm=r.buf(n*4),bv=r.buf(n*4),bpo=r.buf(n*4),bmo=r.buf(n*4),bvo=r.buf(n*4),bn=r.buf(4);
    r.write(bp,p.data(),n*4);r.write(bg,g.data(),n*4);r.write(bm,m.data(),n*4);r.write(bv,v.data(),n*4);
    auto kn=r.kernel("gradnorm_serial");r.arg(kn,0,bg);r.arg(kn,1,bn);r.arg(kn,2,n);r.run1(kn,1);r.read(bn,norm.data(),4);
    add(ts,cmp("gradient_norm_fp32",norm,rnorm,2e-4,2e-5));
    float clip=std::min(1.f,1.f/(rnorm[0]+1e-6f)),lr=1e-4f,b1=.9f,b2=.95f,eps=1e-8f,wd=.1f;
    for(int i=0;i<n;++i){float gg=g[i]*clip;float mn=b1*m[i]+(1-b1)*gg;float vn=b2*v[i]+(1-b2)*gg*gg;rmo[i]=mn;rvo[i]=vn;rpo[i]=p[i]*(1-lr*wd)-lr*(mn/(1-b1))/(std::sqrt(vn/(1-b2))+eps);}
    auto k=r.kernel("adamw_step");r.arg(k,0,bp);r.arg(k,1,bg);r.arg(k,2,bm);r.arg(k,3,bv);r.arg(k,4,bpo);r.arg(k,5,bmo);r.arg(k,6,bvo);r.arg(k,7,n);r.arg(k,8,lr);r.arg(k,9,b1);r.arg(k,10,b2);r.arg(k,11,eps);r.arg(k,12,wd);r.arg(k,13,clip);r.run1(k,n);
    r.read(bpo,po.data(),n*4);r.read(bmo,mo.data(),n*4);r.read(bvo,vo.data(),n*4);
    add(ts,cmp("adamw_parameter_step",po,rpo,3e-6,3e-5));add(ts,cmp("adamw_moment_m",mo,rmo,2e-7,2e-6));add(ts,cmp("adamw_moment_v",vo,rvo,2e-7,2e-6));
}

} // namespace

std::string runNativeOpenClConformanceJson(){
    if(!gCached.empty())return gCached;
    try{
        Runtime& r=rt();Rng q;std::vector<Test>ts;
        testVec(r,ts,q);
        testSilu(r,ts,q);
        testRope(r,ts,q);
        testRms(r,ts,q);
        testGemmShape(r,ts,q,"d384_to_d384",256,384,384,true);
        testGemmShape(r,ts,q,"d384_to_kv128",256,384,128,true);
        testGemmShape(r,ts,q,"d384_to_ff1152",256,384,1152,true);
        testGemmShape(r,ts,q,"ff1152_to_d384",256,1152,384,true);
        testGemmShape(r,ts,q,"d384_to_vocab14000",256,384,14000,true);
        testSoftmax(r,ts,q);
        testGqa(r,ts,q);
        testEmbedding(r,ts,q);
        testCe(r,ts,q);
        testOpt(r,ts,q);

        bool pass=true;for(const auto&t:ts)pass=pass&&t.pass;
        std::ostringstream o;o<<std::setprecision(9);
        o<<"{\"status\":\""<<(pass?"PASS":"FAIL")<<"\""
         <<",\"schema\":\"native_opencl_model0001_conformance_v1\""
         <<",\"backend\":\"PURE_OPENCL_C_1_2_FP32_BUFFER\""
         <<",\"device\":\""<<esc(r.deviceName)<<"\""
         <<",\"opencl_c\":\""<<esc(r.openclC)<<"\""
         <<",\"fast_math\":false"
         <<",\"float_atomics\":false"
         <<",\"images\":false"
         <<",\"mnn_gpu_used\":false"
         <<",\"resource_lifetime\":\"process\""
         <<",\"tests\":[";
        for(size_t i=0;i<ts.size();++i){if(i)o<<",";const auto&t=ts[i];
            o<<"{\"name\":\""<<t.name<<"\",\"pass\":"<<(t.pass?"true":"false")
             <<",\"max_abs\":"<<t.maxAbs<<",\"max_rel\":"<<t.maxRel
             <<",\"abs_tol\":"<<t.absTol<<",\"rel_tol\":"<<t.relTol<<"}";
        }
        o<<"]}";
        gCached=o.str();return gCached;
    }catch(const std::exception&e){
        return std::string("{\"status\":\"FAIL\",\"schema\":\"native_opencl_model0001_conformance_v1\",\"error\":\"")+esc(e.what())+"\"}";
    }
}
} // namespace at
