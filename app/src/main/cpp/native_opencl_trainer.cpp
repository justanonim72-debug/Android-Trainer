#include "native_opencl_trainer.hpp"

#include <CL/cl.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef ANDROID_TRAINER_GIT_COMMIT
#define ANDROID_TRAINER_GIT_COMMIT "local"
#endif

namespace at {
namespace {

constexpr int S = 256;
constexpr int V = 14000;
constexpr int D = 384;
constexpr int HQ = 6;
constexpr int HKV = 2;
constexpr int HD = 64;
constexpr int FF = 1152;
constexpr int LAYERS = 8;
constexpr int TILE = 16;
constexpr int REDUCE_ITEMS = 256;
constexpr int REDUCE_LOCAL = 64;
constexpr int REDUCE_GROUPS = REDUCE_ITEMS / REDUCE_LOCAL;
constexpr int BENCH_STEPS = 20;

void req(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

size_t bytesFor(size_t elements) {
    req(elements <= std::numeric_limits<size_t>::max() / sizeof(float),
        "OpenCL allocation size overflow");
    return elements * sizeof(float);
}

size_t roundUp(size_t n, size_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

double relativeError(double got, double ref) {
    return std::abs(got - ref) /
        std::max({1.0e-12, std::abs(got), std::abs(ref)});
}

const char* kTrainerSource = R"CLC(
#pragma OPENCL FP_CONTRACT OFF

#define TILE 16

__kernel void embedding_lookup(
    __global const int* tokens,
    __global const float* weight,
    __global float* out,
    int rows,
    int d) {
    int g = (int)get_global_id(0);
    int total = rows * d;
    if (g >= total) return;
    int row = g / d;
    int col = g - row * d;
    out[g] = weight[tokens[row] * d + col];
}

__kernel void vector_add(
    __global const float* a,
    __global const float* b,
    __global float* out,
    int n) {
    int i = (int)get_global_id(0);
    if (i < n) out[i] = a[i] + b[i];
}

__kernel void vector_add_inplace(
    __global float* dst,
    __global const float* src,
    int n) {
    int i = (int)get_global_id(0);
    if (i < n) dst[i] += src[i];
}

__kernel void rmsnorm_forward(
    __global const float* x,
    __global const float* w,
    __global float* y,
    int rows,
    int d,
    float eps) {
    int row = (int)get_global_id(0);
    if (row >= rows) return;
    int base = row * d;
    float ss = 0.0f;
    for (int j = 0; j < d; ++j) {
        float z = x[base + j];
        ss += z * z;
    }
    float inv = rsqrt(ss / (float)d + eps);
    for (int j = 0; j < d; ++j) y[base + j] = x[base + j] * inv * w[j];
}

__kernel void rmsnorm_dx(
    __global const float* x,
    __global const float* w,
    __global const float* dy,
    __global float* dx,
    __global float* inv_rows,
    int rows,
    int d,
    float eps) {
    int row = (int)get_global_id(0);
    if (row >= rows) return;
    int base = row * d;
    float ss = 0.0f;
    for (int j = 0; j < d; ++j) {
        float z = x[base + j];
        ss += z * z;
    }
    float inv = rsqrt(ss / (float)d + eps);
    inv_rows[row] = inv;
    float dot = 0.0f;
    for (int j = 0; j < d; ++j) dot += dy[base + j] * w[j] * x[base + j];
    float corr = dot * (inv * inv * inv) / (float)d;
    for (int j = 0; j < d; ++j) {
        dx[base + j] = dy[base + j] * w[j] * inv - x[base + j] * corr;
    }
}

__kernel void rmsnorm_dw(
    __global const float* x,
    __global const float* dy,
    __global const float* inv_rows,
    __global float* dw,
    int rows,
    int d) {
    int col = (int)get_global_id(0);
    if (col >= d) return;
    float acc = 0.0f;
    for (int row = 0; row < rows; ++row) {
        int base = row * d;
        acc += dy[base + col] * x[base + col] * inv_rows[row];
    }
    dw[col] = acc;
}

// C[M,N] = A[M,K] * W[N,K]^T.
__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void linear_forward(
    __global const float* a,
    __global const float* w,
    __global float* c,
    int m,
    int k,
    int n) {
    __local float aa[TILE][TILE];
    __local float ww[TILE][TILE];

    const int lx = (int)get_local_id(0);
    const int ly = (int)get_local_id(1);
    const int tx = lx * 2;
    const int ty = ly * 2;
    const int n0 = (int)get_group_id(0) * TILE + tx;
    const int n1 = n0 + 1;
    const int m0 = (int)get_group_id(1) * TILE + ty;
    const int m1 = m0 + 1;

    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;

    for (int t = 0; t < k; t += TILE) {
        const int kx0 = t + tx;
        const int kx1 = kx0 + 1;
        const int ky0 = t + ty;
        const int ky1 = ky0 + 1;

        aa[ty][tx] =
            (m0 < m && kx0 < k) ? a[m0 * k + kx0] : 0.0f;
        aa[ty][tx + 1] =
            (m0 < m && kx1 < k) ? a[m0 * k + kx1] : 0.0f;
        aa[ty + 1][tx] =
            (m1 < m && kx0 < k) ? a[m1 * k + kx0] : 0.0f;
        aa[ty + 1][tx + 1] =
            (m1 < m && kx1 < k) ? a[m1 * k + kx1] : 0.0f;

        ww[ty][tx] =
            (n0 < n && ky0 < k) ? w[n0 * k + ky0] : 0.0f;
        ww[ty][tx + 1] =
            (n1 < n && ky0 < k) ? w[n1 * k + ky0] : 0.0f;
        ww[ty + 1][tx] =
            (n0 < n && ky1 < k) ? w[n0 * k + ky1] : 0.0f;
        ww[ty + 1][tx + 1] =
            (n1 < n && ky1 < k) ? w[n1 * k + ky1] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int q = 0; q < TILE; ++q) {
            const float a0 = aa[ty][q];
            const float a1 = aa[ty + 1][q];
            const float w0 = ww[q][tx];
            const float w1 = ww[q][tx + 1];
            acc00 += a0 * w0;
            acc01 += a0 * w1;
            acc10 += a1 * w0;
            acc11 += a1 * w1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (m0 < m && n0 < n) c[m0 * n + n0] = acc00;
    if (m0 < m && n1 < n) c[m0 * n + n1] = acc01;
    if (m1 < m && n0 < n) c[m1 * n + n0] = acc10;
    if (m1 < m && n1 < n) c[m1 * n + n1] = acc11;
}

// DX[M,K] = DY[M,N] * W[N,K].
__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void linear_dinput(
    __global const float* dy,
    __global const float* w,
    __global float* dx,
    int m,
    int k,
    int n) {
    __local float yy[TILE][TILE];
    __local float ww[TILE][TILE];

    const int lx = (int)get_local_id(0);
    const int ly = (int)get_local_id(1);
    const int tx = lx * 2;
    const int ty = ly * 2;
    const int k0 = (int)get_group_id(0) * TILE + tx;
    const int k1 = k0 + 1;
    const int m0 = (int)get_group_id(1) * TILE + ty;
    const int m1 = m0 + 1;

    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;

    for (int t = 0; t < n; t += TILE) {
        const int n0 = t + tx;
        const int n1 = n0 + 1;
        const int nr0 = t + ty;
        const int nr1 = nr0 + 1;

        yy[ty][tx] =
            (m0 < m && n0 < n) ? dy[m0 * n + n0] : 0.0f;
        yy[ty][tx + 1] =
            (m0 < m && n1 < n) ? dy[m0 * n + n1] : 0.0f;
        yy[ty + 1][tx] =
            (m1 < m && n0 < n) ? dy[m1 * n + n0] : 0.0f;
        yy[ty + 1][tx + 1] =
            (m1 < m && n1 < n) ? dy[m1 * n + n1] : 0.0f;

        ww[ty][tx] =
            (nr0 < n && k0 < k) ? w[nr0 * k + k0] : 0.0f;
        ww[ty][tx + 1] =
            (nr0 < n && k1 < k) ? w[nr0 * k + k1] : 0.0f;
        ww[ty + 1][tx] =
            (nr1 < n && k0 < k) ? w[nr1 * k + k0] : 0.0f;
        ww[ty + 1][tx + 1] =
            (nr1 < n && k1 < k) ? w[nr1 * k + k1] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int q = 0; q < TILE; ++q) {
            const float y0 = yy[ty][q];
            const float y1 = yy[ty + 1][q];
            const float w0 = ww[q][tx];
            const float w1 = ww[q][tx + 1];
            acc00 += y0 * w0;
            acc01 += y0 * w1;
            acc10 += y1 * w0;
            acc11 += y1 * w1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (m0 < m && k0 < k) dx[m0 * k + k0] = acc00;
    if (m0 < m && k1 < k) dx[m0 * k + k1] = acc01;
    if (m1 < m && k0 < k) dx[m1 * k + k0] = acc10;
    if (m1 < m && k1 < k) dx[m1 * k + k1] = acc11;
}

// DW[N,K] = DY[M,N]^T * A[M,K]. One work-item owns every output element,
// so the accumulation is deterministic and race-free.
__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void linear_dweight(
    __global const float* dy,
    __global const float* a,
    __global float* dw,
    int m,
    int k,
    int n) {
    __local float yy[TILE][TILE];
    __local float aa[TILE][TILE];

    const int lx = (int)get_local_id(0);
    const int ly = (int)get_local_id(1);
    const int tx = lx * 2;
    const int ty = ly * 2;
    const int k0 = (int)get_group_id(0) * TILE + tx;
    const int k1 = k0 + 1;
    const int n0 = (int)get_group_id(1) * TILE + ty;
    const int n1 = n0 + 1;

    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;

    for (int t = 0; t < m; t += TILE) {
        // yy is stored transposed as [output-n][reduction-m], matching the
        // original 16x16 kernel exactly. aa is [reduction-m][output-k].
        const int mc0 = t + tx;
        const int mc1 = mc0 + 1;
        const int mr0 = t + ty;
        const int mr1 = mr0 + 1;

        yy[ty][tx] =
            (mc0 < m && n0 < n) ? dy[mc0 * n + n0] : 0.0f;
        yy[ty][tx + 1] =
            (mc1 < m && n0 < n) ? dy[mc1 * n + n0] : 0.0f;
        yy[ty + 1][tx] =
            (mc0 < m && n1 < n) ? dy[mc0 * n + n1] : 0.0f;
        yy[ty + 1][tx + 1] =
            (mc1 < m && n1 < n) ? dy[mc1 * n + n1] : 0.0f;

        aa[ty][tx] =
            (mr0 < m && k0 < k) ? a[mr0 * k + k0] : 0.0f;
        aa[ty][tx + 1] =
            (mr0 < m && k1 < k) ? a[mr0 * k + k1] : 0.0f;
        aa[ty + 1][tx] =
            (mr1 < m && k0 < k) ? a[mr1 * k + k0] : 0.0f;
        aa[ty + 1][tx + 1] =
            (mr1 < m && k1 < k) ? a[mr1 * k + k1] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int q = 0; q < TILE; ++q) {
            const float y0 = yy[ty][q];
            const float y1 = yy[ty + 1][q];
            const float a0 = aa[q][tx];
            const float a1 = aa[q][tx + 1];
            acc00 += y0 * a0;
            acc01 += y0 * a1;
            acc10 += y1 * a0;
            acc11 += y1 * a1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (n0 < n && k0 < k) dw[n0 * k + k0] = acc00;
    if (n0 < n && k1 < k) dw[n0 * k + k1] = acc01;
    if (n1 < n && k0 < k) dw[n1 * k + k0] = acc10;
    if (n1 < n && k1 < k) dw[n1 * k + k1] = acc11;
}

__kernel void split_heads(
    __global const float* flat,
    __global float* headsOut,
    int rows,
    int heads,
    int hd) {
    int g = (int)get_global_id(0);
    int total = rows * heads * hd;
    if (g >= total) return;
    int d = g % hd;
    int t = g / hd;
    int h = t % heads;
    int row = t / heads;
    headsOut[(h * rows + row) * hd + d] = flat[(row * heads + h) * hd + d];
}

__kernel void merge_heads(
    __global const float* headsIn,
    __global float* flat,
    int rows,
    int heads,
    int hd) {
    int g = (int)get_global_id(0);
    int total = rows * heads * hd;
    if (g >= total) return;
    int d = g % hd;
    int t = g / hd;
    int h = t % heads;
    int row = t / heads;
    flat[(row * heads + h) * hd + d] = headsIn[(h * rows + row) * hd + d];
}

__kernel void rope_forward(
    __global const float* x,
    __global const float* cs,
    __global const float* sn,
    __global float* y,
    int heads,
    int rows,
    int hd) {
    int g = (int)get_global_id(0);
    int pairs = heads * rows * (hd / 2);
    if (g >= pairs) return;
    int pair = g % (hd / 2);
    int t = g / (hd / 2);
    int row = t % rows;
    int h = t / rows;
    int base = (h * rows + row) * hd + pair * 2;
    int rb = row * hd + pair * 2;
    float c = cs[rb], s = sn[rb];
    float a = x[base], b = x[base + 1];
    y[base] = a * c - b * s;
    y[base + 1] = b * c + a * s;
}

__kernel void rope_backward(
    __global const float* dy,
    __global const float* cs,
    __global const float* sn,
    __global float* dx,
    int heads,
    int rows,
    int hd) {
    int g = (int)get_global_id(0);
    int pairs = heads * rows * (hd / 2);
    if (g >= pairs) return;
    int pair = g % (hd / 2);
    int t = g / (hd / 2);
    int row = t % rows;
    int h = t / rows;
    int base = (h * rows + row) * hd + pair * 2;
    int rb = row * hd + pair * 2;
    float c = cs[rb], s = sn[rb];
    float u = dy[base], v = dy[base + 1];
    dx[base] = u * c + v * s;
    dx[base + 1] = -u * s + v * c;
}

__kernel void repeat_kv(
    __global const float* src,
    __global float* dst,
    int hq,
    int hkv,
    int rows,
    int hd) {
    int g = (int)get_global_id(0);
    int total = hq * rows * hd;
    if (g >= total) return;
    int d = g % hd;
    int t = g / hd;
    int row = t % rows;
    int qh = t / rows;
    int kv = qh / (hq / hkv);
    dst[g] = src[(kv * rows + row) * hd + d];
}

__kernel void reduce_gqa(
    __global const float* src,
    __global float* dst,
    int hq,
    int hkv,
    int rows,
    int hd) {
    int g = (int)get_global_id(0);
    int total = hkv * rows * hd;
    if (g >= total) return;
    int d = g % hd;
    int t = g / hd;
    int row = t % rows;
    int kv = t / rows;
    int repeat = hq / hkv;
    float acc = 0.0f;
    for (int r = 0; r < repeat; ++r) {
        int qh = kv * repeat + r;
        acc += src[(qh * rows + row) * hd + d];
    }
    dst[g] = acc;
}

// C[B,M,N] = A[B,M,K] * Bv[B,N,K]^T.
__kernel void bmm_nt(
    __global const float* a,
    __global const float* bv,
    __global float* c,
    int batches,
    int m,
    int n,
    int k,
    float scale) {
    int nn = (int)get_global_id(0);
    int br = (int)get_global_id(1);
    if (nn >= n || br >= batches * m) return;
    int b = br / m, mm = br - b * m;
    int ab = (b * m + mm) * k;
    int bb = (b * n + nn) * k;
    float acc = 0.0f;
    for (int q = 0; q < k; ++q) acc += a[ab + q] * bv[bb + q];
    c[(b * m + mm) * n + nn] = acc * scale;
}

// C[B,M,K] = A[B,M,N] * Bv[B,N,K].
__kernel void bmm_nn(
    __global const float* a,
    __global const float* bv,
    __global float* c,
    int batches,
    int m,
    int n,
    int k,
    float scale) {
    int kk = (int)get_global_id(0);
    int br = (int)get_global_id(1);
    if (kk >= k || br >= batches * m) return;
    int b = br / m, mm = br - b * m;
    int ab = (b * m + mm) * n;
    float acc = 0.0f;
    for (int q = 0; q < n; ++q) acc += a[ab + q] * bv[(b * n + q) * k + kk];
    c[(b * m + mm) * k + kk] = acc * scale;
}

// C[B,N,K] = A[B,M,N]^T * Bv[B,M,K].
__kernel void bmm_left_t(
    __global const float* a,
    __global const float* bv,
    __global float* c,
    int batches,
    int m,
    int n,
    int k,
    float scale) {
    int kk = (int)get_global_id(0);
    int bn = (int)get_global_id(1);
    if (kk >= k || bn >= batches * n) return;
    int b = bn / n, nn = bn - b * n;
    float acc = 0.0f;
    for (int q = 0; q < m; ++q) {
        acc += a[(b * m + q) * n + nn] * bv[(b * m + q) * k + kk];
    }
    c[(b * n + nn) * k + kk] = acc * scale;
}

__kernel void causal_softmax_forward(
    __global const float* scores,
    __global float* probability,
    int totalRows,
    int seq) {
    int row = (int)get_global_id(0);
    if (row >= totalRows) return;
    int pos = row % seq;
    int base = row * seq;
    float mx = -3.402823466e+38F;
    for (int j = 0; j <= pos; ++j) mx = fmax(mx, scores[base + j]);
    float den = 0.0f;
    for (int j = 0; j <= pos; ++j) den += exp(scores[base + j] - mx);
    for (int j = 0; j < seq; ++j) {
        probability[base + j] = j <= pos ? exp(scores[base + j] - mx) / den : 0.0f;
    }
}

__kernel void causal_softmax_backward(
    __global const float* probability,
    __global float* gradient,
    int totalRows,
    int seq) {
    int row = (int)get_global_id(0);
    if (row >= totalRows) return;
    int pos = row % seq;
    int base = row * seq;
    float dot = 0.0f;
    for (int j = 0; j <= pos; ++j) dot += probability[base + j] * gradient[base + j];
    for (int j = 0; j < seq; ++j) {
        gradient[base + j] = j <= pos
            ? probability[base + j] * (gradient[base + j] - dot)
            : 0.0f;
    }
}

__kernel void silu_multiply(
    __global const float* gate,
    __global const float* up,
    __global float* out,
    int n) {
    int i = (int)get_global_id(0);
    if (i >= n) return;
    float s = 1.0f / (1.0f + exp(-gate[i]));
    out[i] = gate[i] * s * up[i];
}

__kernel void silu_multiply_backward(
    __global const float* gate,
    __global const float* up,
    __global const float* dout,
    __global float* dgate,
    __global float* dup,
    int n) {
    int i = (int)get_global_id(0);
    if (i >= n) return;
    float s = 1.0f / (1.0f + exp(-gate[i]));
    float silu = gate[i] * s;
    dgate[i] = dout[i] * up[i] * (s + gate[i] * s * (1.0f - s));
    dup[i] = dout[i] * silu;
}

__kernel void cross_entropy_forward_backward(
    __global const float* logits,
    __global const int* targets,
    __global float* rowLoss,
    __global float* dlogits,
    int rows,
    int vocab) {
    int row = (int)get_global_id(0);
    if (row >= rows) return;
    int base = row * vocab;
    float mx = -3.402823466e+38F;
    for (int j = 0; j < vocab; ++j) mx = fmax(mx, logits[base + j]);
    float den = 0.0f;
    for (int j = 0; j < vocab; ++j) den += exp(logits[base + j] - mx);
    rowLoss[row] = -(logits[base + targets[row]] - mx - log(den));
    float inv = 1.0f / den;
    float mean = 1.0f / (float)rows;
    for (int j = 0; j < vocab; ++j) {
        float p = exp(logits[base + j] - mx) * inv;
        dlogits[base + j] = (p - (j == targets[row] ? 1.0f : 0.0f)) * mean;
    }
}

__kernel void mean_rows(
    __global const float* rows,
    __global float* out,
    int n) {
    if (get_global_id(0) != 0) return;
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += rows[i];
    out[0] = acc / (float)n;
}

__kernel void embedding_add(
    __global float* embeddingGrad,
    __global const float* lookupGrad,
    __global const int* tokenIds,
    __global const int* positions,
    __global const int* offsets,
    int uniqueCount,
    int d) {
    int g = (int)get_global_id(0);
    int total = uniqueCount * d;
    if (g >= total) return;
    int u = g / d, col = g - u * d;
    float acc = 0.0f;
    for (int q = offsets[u]; q < offsets[u + 1]; ++q) {
        acc += lookupGrad[positions[q] * d + col];
    }
    embeddingGrad[tokenIds[u] * d + col] += acc;
}

__kernel void sumsq_partial(
    __global const float* values,
    __global float* partial,
    int n,
    int outputOffset) {
    __local float localSum[64];
    int lid = (int)get_local_id(0);
    int gid = (int)get_global_id(0);
    int stride = (int)get_global_size(0);
    float acc = 0.0f;
    for (int i = gid; i < n; i += stride) acc += values[i] * values[i];
    localSum[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int step = 32; step > 0; step >>= 1) {
        if (lid < step) localSum[lid] += localSum[lid + step];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial[outputOffset + (int)get_group_id(0)] = localSum[0];
}

__kernel void finish_norm(
    __global const float* partial,
    __global float* norm,
    int count) {
    if (get_global_id(0) != 0) return;
    float acc = 0.0f;
    for (int i = 0; i < count; ++i) acc += partial[i];
    norm[0] = sqrt(acc);
}

__kernel void adamw_update(
    __global float* parameter,
    __global const float* gradient,
    __global float* moment1,
    __global float* moment2,
    __global const float* globalNorm,
    int n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weightDecay,
    float biasCorrection1,
    float biasCorrection2) {
    int i = (int)get_global_id(0);
    if (i >= n) return;
    float clip = fmin(1.0f, 1.0f / (globalNorm[0] + 1.0e-6f));
    float g = gradient[i] * clip;
    float m = beta1 * moment1[i] + (1.0f - beta1) * g;
    float v = beta2 * moment2[i] + (1.0f - beta2) * g * g;
    float update = (m / biasCorrection1) /
        (sqrt(v / biasCorrection2) + eps);
    parameter[i] = parameter[i] * (1.0f - lr * weightDecay) - lr * update;
    moment1[i] = m;
    moment2[i] = v;
}

__kernel void gather_values(
    __global const float* source,
    __global const int* indices,
    __global float* output,
    int count) {
    int i = (int)get_global_id(0);
    if (i < count) output[i] = source[indices[i]];
}
)CLC";

struct OpenClApi {
    void* library = nullptr;
    std::string libraryName;
    decltype(&clGetPlatformIDs) GetPlatformIDs = nullptr;
    decltype(&clGetDeviceIDs) GetDeviceIDs = nullptr;
    decltype(&clGetDeviceInfo) GetDeviceInfo = nullptr;
    decltype(&clCreateContext) CreateContext = nullptr;
    decltype(&clCreateCommandQueue) CreateCommandQueue = nullptr;
    decltype(&clCreateProgramWithSource) CreateProgramWithSource = nullptr;
    decltype(&clBuildProgram) BuildProgram = nullptr;
    decltype(&clGetProgramBuildInfo) GetProgramBuildInfo = nullptr;
    decltype(&clCreateKernel) CreateKernel = nullptr;
    decltype(&clCreateBuffer) CreateBuffer = nullptr;
    decltype(&clSetKernelArg) SetKernelArg = nullptr;
    decltype(&clEnqueueWriteBuffer) EnqueueWriteBuffer = nullptr;
    decltype(&clEnqueueReadBuffer) EnqueueReadBuffer = nullptr;
    decltype(&clEnqueueFillBuffer) EnqueueFillBuffer = nullptr;
    decltype(&clEnqueueNDRangeKernel) EnqueueNDRangeKernel = nullptr;
    decltype(&clGetEventProfilingInfo) GetEventProfilingInfo = nullptr;
    decltype(&clFinish) Finish = nullptr;

    template <class T>
    void symbol(T& destination, const char* name) {
        destination = reinterpret_cast<T>(dlsym(library, name));
        req(destination != nullptr, std::string("missing OpenCL symbol ") + name);
    }

    void load() {
        if (library) return;
        const char* candidates[] = {"libOpenCL.so", "libGLES_mali.so", "libmali.so"};
        for (const char* candidate : candidates) {
            library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
            if (library) {
                libraryName = candidate;
                break;
            }
        }
        req(library != nullptr, "cannot dlopen Android OpenCL library");
        symbol(GetPlatformIDs, "clGetPlatformIDs");
        symbol(GetDeviceIDs, "clGetDeviceIDs");
        symbol(GetDeviceInfo, "clGetDeviceInfo");
        symbol(CreateContext, "clCreateContext");
        symbol(CreateCommandQueue, "clCreateCommandQueue");
        symbol(CreateProgramWithSource, "clCreateProgramWithSource");
        symbol(BuildProgram, "clBuildProgram");
        symbol(GetProgramBuildInfo, "clGetProgramBuildInfo");
        symbol(CreateKernel, "clCreateKernel");
        symbol(CreateBuffer, "clCreateBuffer");
        symbol(SetKernelArg, "clSetKernelArg");
        symbol(EnqueueWriteBuffer, "clEnqueueWriteBuffer");
        symbol(EnqueueReadBuffer, "clEnqueueReadBuffer");
        symbol(EnqueueFillBuffer, "clEnqueueFillBuffer");
        symbol(EnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
        symbol(GetEventProfilingInfo, "clGetEventProfilingInfo");
        symbol(Finish, "clFinish");
    }
};

struct ProcessRuntime {
    OpenClApi api;
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_command_queue profilingQueue = nullptr;
    cl_program program = nullptr;
    std::map<std::string, cl_kernel> kernels;

    struct ProfiledEvent {
        std::string kernel;
        cl_event event = nullptr;
    };
    bool diagnosticQueueActive = false;
    bool kernelCaptureActive = false;
    std::vector<ProfiledEvent> profiledEvents;
    std::string deviceName;
    std::string vendor;
    std::string deviceVersion;
    std::string driverVersion;
    std::string openclCVersion;
    cl_ulong globalMemory = 0;
    cl_ulong maxAllocation = 0;
    size_t localMemory = 0;
    size_t maxWorkGroup = 0;
    bool initialized = false;

    std::string deviceString(cl_device_info key) {
        size_t size = 0;
        cl_int ec = api.GetDeviceInfo(device, key, 0, nullptr, &size);
        req(ec == CL_SUCCESS && size > 0,
            "clGetDeviceInfo string failed ec=" + std::to_string(ec));
        std::string value(size, '\0');
        ec = api.GetDeviceInfo(device, key, size, &value[0], nullptr);
        req(ec == CL_SUCCESS,
            "clGetDeviceInfo string read failed ec=" + std::to_string(ec));
        while (!value.empty() && value.back() == '\0') value.pop_back();
        return value;
    }

    template <class T>
    void deviceValue(cl_device_info key, T* value) {
        cl_int ec = api.GetDeviceInfo(device, key, sizeof(T), value, nullptr);
        req(ec == CL_SUCCESS,
            "clGetDeviceInfo value failed ec=" + std::to_string(ec));
    }

    void initialize() {
        if (initialized) return;
        api.load();
        cl_uint platformCount = 0;
        cl_int ec = api.GetPlatformIDs(0, nullptr, &platformCount);
        req(ec == CL_SUCCESS && platformCount > 0, "no OpenCL platform");
        std::vector<cl_platform_id> platforms(platformCount);
        ec = api.GetPlatformIDs(platformCount, platforms.data(), nullptr);
        req(ec == CL_SUCCESS, "OpenCL platform enumeration failed");
        for (cl_platform_id candidate : platforms) {
            cl_uint count = 0;
            cl_device_id candidateDevice = nullptr;
            ec = api.GetDeviceIDs(
                candidate, CL_DEVICE_TYPE_GPU, 1, &candidateDevice, &count);
            if (ec == CL_SUCCESS && count > 0) {
                platform = candidate;
                device = candidateDevice;
                break;
            }
        }
        req(device != nullptr, "no OpenCL GPU device");

        deviceName = deviceString(CL_DEVICE_NAME);
        vendor = deviceString(CL_DEVICE_VENDOR);
        deviceVersion = deviceString(CL_DEVICE_VERSION);
        driverVersion = deviceString(CL_DRIVER_VERSION);
        openclCVersion = deviceString(CL_DEVICE_OPENCL_C_VERSION);
        deviceValue(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemory);
        deviceValue(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxAllocation);
        cl_ulong local = 0;
        deviceValue(CL_DEVICE_LOCAL_MEM_SIZE, &local);
        localMemory = static_cast<size_t>(local);
        deviceValue(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroup);
        req(maxWorkGroup >= TILE * TILE,
            "OpenCL device work-group limit is below 16x16");

        const cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(platform),
            0,
        };
        context = api.CreateContext(properties, 1, &device, nullptr, nullptr, &ec);
        req(ec == CL_SUCCESS && context != nullptr,
            "clCreateContext failed ec=" + std::to_string(ec));
        queue = api.CreateCommandQueue(context, device, 0, &ec);
        req(ec == CL_SUCCESS && queue != nullptr,
            "clCreateCommandQueue failed ec=" + std::to_string(ec));

        profilingQueue = api.CreateCommandQueue(
            context, device, CL_QUEUE_PROFILING_ENABLE, &ec);
        req(ec == CL_SUCCESS && profilingQueue != nullptr,
            "clCreateCommandQueue profiling failed ec=" + std::to_string(ec));

        const char* source = kTrainerSource;
        const size_t sourceLength = std::strlen(source);
        program = api.CreateProgramWithSource(
            context, 1, &source, &sourceLength, &ec);
        req(ec == CL_SUCCESS && program != nullptr,
            "clCreateProgramWithSource failed ec=" + std::to_string(ec));
        // This is deliberately the complete option list. In particular, no
        // fast-math, relaxed-math, FP16, or vendor-specific option is allowed.
        ec = api.BuildProgram(
            program, 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
        if (ec != CL_SUCCESS) {
            size_t size = 0;
            api.GetProgramBuildInfo(
                program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::string log(size ? size : 1, '\0');
            if (size) {
                api.GetProgramBuildInfo(
                    program, device, CL_PROGRAM_BUILD_LOG, size, &log[0], nullptr);
            }
            throw std::runtime_error(
                "native trainer OpenCL build failed ec=" +
                std::to_string(ec) + " log=" + log);
        }

        static const char* names[] = {
            "embedding_lookup", "vector_add", "vector_add_inplace",
            "rmsnorm_forward", "rmsnorm_dx", "rmsnorm_dw",
            "linear_forward", "linear_dinput", "linear_dweight",
            "split_heads", "merge_heads", "rope_forward", "rope_backward",
            "repeat_kv", "reduce_gqa", "bmm_nt", "bmm_nn", "bmm_left_t",
            "causal_softmax_forward", "causal_softmax_backward",
            "silu_multiply", "silu_multiply_backward",
            "cross_entropy_forward_backward", "mean_rows", "embedding_add",
            "sumsq_partial", "finish_norm", "adamw_update", "gather_values",
        };
        for (const char* name : names) {
            cl_kernel kernel = api.CreateKernel(program, name, &ec);
            req(ec == CL_SUCCESS && kernel != nullptr,
                std::string("clCreateKernel failed for ") + name +
                " ec=" + std::to_string(ec));
            kernels.emplace(name, kernel);
        }
        initialized = true;
    }

    cl_kernel kernel(const char* name) {
        auto it = kernels.find(name);
        req(it != kernels.end(), std::string("unknown kernel ") + name);
        return it->second;
    }

    cl_command_queue activeQueue() const {
        return diagnosticQueueActive ? profilingQueue : queue;
    }

    std::string kernelName(cl_kernel value) const {
        for (const auto& pair : kernels) {
            if (pair.second == value) return pair.first;
        }
        return "unknown";
    }

    void beginDiagnosticQueue() {
        req(!kernelCaptureActive, "cannot switch diagnostic queue during capture");
        diagnosticQueueActive = true;
    }

    void endDiagnosticQueue() {
        req(!kernelCaptureActive, "cannot leave diagnostic queue during capture");
        diagnosticQueueActive = false;
    }

    void beginKernelCapture() {
        req(diagnosticQueueActive,
            "kernel capture requires profiling-enabled diagnostic queue");
        req(!kernelCaptureActive, "kernel capture already active");
        profiledEvents.clear();
        kernelCaptureActive = true;
    }

    std::map<std::string, std::pair<int, double>> endKernelCapture() {
        req(kernelCaptureActive, "kernel capture was not active");
        finish();
        kernelCaptureActive = false;
        std::map<std::string, std::pair<int, double>> totals;
        for (const auto& item : profiledEvents) {
            req(item.event != nullptr, "null OpenCL profiling event");
            cl_ulong start = 0, end = 0;
            cl_int ec = api.GetEventProfilingInfo(
                item.event, CL_PROFILING_COMMAND_START,
                sizeof(start), &start, nullptr);
            req(ec == CL_SUCCESS,
                "clGetEventProfilingInfo(start) failed ec=" + std::to_string(ec));
            ec = api.GetEventProfilingInfo(
                item.event, CL_PROFILING_COMMAND_END,
                sizeof(end), &end, nullptr);
            req(ec == CL_SUCCESS && end >= start,
                "clGetEventProfilingInfo(end) failed ec=" + std::to_string(ec));
            auto& aggregate = totals[item.kernel];
            aggregate.first += 1;
            aggregate.second += static_cast<double>(end - start) * 1.0e-9;
        }
        // Deliberately do not clReleaseEvent here. This process-long diagnostic
        // path follows the existing Mali no-teardown policy and runs once.
        return totals;
    }

    cl_mem allocate(size_t bytes, cl_mem_flags flags = CL_MEM_READ_WRITE) {
        req(bytes > 0 && static_cast<cl_ulong>(bytes) <= maxAllocation,
            "OpenCL buffer exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE: " +
            std::to_string(bytes));
        cl_int ec = CL_SUCCESS;
        cl_mem value = api.CreateBuffer(context, flags, bytes, nullptr, &ec);
        req(ec == CL_SUCCESS && value != nullptr,
            "clCreateBuffer failed bytes=" + std::to_string(bytes) +
            " ec=" + std::to_string(ec));
        return value;
    }

    void write(cl_mem buffer, const void* data, size_t bytes, size_t offset = 0) {
        cl_int ec = api.EnqueueWriteBuffer(
            activeQueue(), buffer, CL_TRUE, offset, bytes, data, 0, nullptr, nullptr);
        req(ec == CL_SUCCESS,
            "clEnqueueWriteBuffer failed ec=" + std::to_string(ec));
    }

    void read(cl_mem buffer, void* data, size_t bytes, size_t offset = 0) {
        cl_int ec = api.EnqueueReadBuffer(
            activeQueue(), buffer, CL_TRUE, offset, bytes, data, 0, nullptr, nullptr);
        req(ec == CL_SUCCESS,
            "clEnqueueReadBuffer failed ec=" + std::to_string(ec));
    }

    void zero(cl_mem buffer, size_t bytes) {
        const float value = 0.0f;
        cl_int ec = api.EnqueueFillBuffer(
            activeQueue(), buffer, &value, sizeof(value), 0, bytes, 0, nullptr, nullptr);
        req(ec == CL_SUCCESS,
            "clEnqueueFillBuffer failed ec=" + std::to_string(ec));
    }

    template <class T>
    void argument(cl_kernel kernelValue, cl_uint index, const T& value) {
        cl_int ec = api.SetKernelArg(
            kernelValue, index, sizeof(T), static_cast<const void*>(&value));
        req(ec == CL_SUCCESS,
            "clSetKernelArg failed index=" + std::to_string(index) +
            " ec=" + std::to_string(ec));
    }

    void enqueue1(cl_kernel kernelValue, size_t count, size_t local = 64) {
        req(local > 0 && local <= maxWorkGroup, "invalid 1-D local size");
        const size_t global = roundUp(std::max<size_t>(count, 1), local);
        cl_event event = nullptr;
        cl_event* eventOut = kernelCaptureActive ? &event : nullptr;
        cl_int ec = api.EnqueueNDRangeKernel(
            activeQueue(), kernelValue, 1, nullptr, &global, &local,
            0, nullptr, eventOut);
        req(ec == CL_SUCCESS,
            "clEnqueueNDRangeKernel(1D) failed ec=" + std::to_string(ec));
        if (kernelCaptureActive) {
            req(event != nullptr, "OpenCL 1D profiling event missing");
            profiledEvents.push_back({kernelName(kernelValue), event});
        }
    }

    void enqueue2(cl_kernel kernelValue, size_t x, size_t y) {
        const size_t global[2] = {roundUp(x, TILE), roundUp(y, TILE)};
        const size_t local[2] = {TILE, TILE};
        cl_event event = nullptr;
        cl_event* eventOut = kernelCaptureActive ? &event : nullptr;
        cl_int ec = api.EnqueueNDRangeKernel(
            activeQueue(), kernelValue, 2, nullptr, global, local,
            0, nullptr, eventOut);
        req(ec == CL_SUCCESS,
            "clEnqueueNDRangeKernel(2D) failed ec=" + std::to_string(ec));
        if (kernelCaptureActive) {
            req(event != nullptr, "OpenCL 2D profiling event missing");
            profiledEvents.push_back({kernelName(kernelValue), event});
        }
    }

    void enqueue2Micro2x2(cl_kernel kernelValue, size_t x, size_t y) {
        constexpr size_t MICRO = 2;
        constexpr size_t LOCAL = 8;
        const size_t gx = (x + MICRO - 1) / MICRO;
        const size_t gy = (y + MICRO - 1) / MICRO;
        const size_t global[2] = {roundUp(gx, LOCAL), roundUp(gy, LOCAL)};
        const size_t local[2] = {LOCAL, LOCAL};
        cl_event event = nullptr;
        cl_event* eventOut = kernelCaptureActive ? &event : nullptr;
        cl_int ec = api.EnqueueNDRangeKernel(
            activeQueue(), kernelValue, 2, nullptr, global, local,
            0, nullptr, eventOut);
        req(ec == CL_SUCCESS,
            "clEnqueueNDRangeKernel(2D micro2x2) failed ec=" +
            std::to_string(ec));
        if (kernelCaptureActive) {
            req(event != nullptr, "OpenCL micro2x2 profiling event missing");
            profiledEvents.push_back({kernelName(kernelValue), event});
        }
    }

    void finish() {
        cl_int ec = api.Finish(activeQueue());
        req(ec == CL_SUCCESS, "clFinish failed ec=" + std::to_string(ec));
    }

    std::string json() const {
        std::ostringstream out;
        out << "{\"library\":\"" << jsonEscape(api.libraryName)
            << "\",\"name\":\"" << jsonEscape(deviceName)
            << "\",\"vendor\":\"" << jsonEscape(vendor)
            << "\",\"device_version\":\"" << jsonEscape(deviceVersion)
            << "\",\"driver_version\":\"" << jsonEscape(driverVersion)
            << "\",\"opencl_c_version\":\"" << jsonEscape(openclCVersion)
            << "\",\"global_mem_bytes\":"
            << static_cast<unsigned long long>(globalMemory)
            << ",\"max_alloc_bytes\":"
            << static_cast<unsigned long long>(maxAllocation)
            << ",\"local_mem_bytes\":" << localMemory
            << ",\"max_work_group_size\":" << maxWorkGroup << "}";
        return out.str();
    }
};

// Intentionally process-long. No clRelease* function is loaded or called, and
// the shared context/queue/program/kernel set is never recreated per step.
ProcessRuntime& processRuntime() {
    static ProcessRuntime* runtime = new ProcessRuntime();
    runtime->initialize();
    return *runtime;
}

struct SlotBuffers {
    std::string name;
    std::vector<int> shape;
    size_t elements = 0;
    float weightDecay = 0.0f;
    cl_mem parameter = nullptr;
    cl_mem gradient = nullptr;
    cl_mem moment1 = nullptr;
    cl_mem moment2 = nullptr;
};

struct LayerBuffers {
    SlotBuffers* attnNorm = nullptr;
    SlotBuffers* q = nullptr;
    SlotBuffers* k = nullptr;
    SlotBuffers* v = nullptr;
    SlotBuffers* o = nullptr;
    SlotBuffers* ffnNorm = nullptr;
    SlotBuffers* gate = nullptr;
    SlotBuffers* up = nullptr;
    SlotBuffers* down = nullptr;
};

struct ProbeError {
    double maxAbs = 0.0;
    double maxRel = 0.0;
    std::string slot;
    int index = -1;
    double reference = 0.0;
    double got = 0.0;
};

struct ForwardGate {
    bool pass = false;
    double loss = 0.0;
    double lossAbs = 0.0;
    ProbeError logits;
    int position = -1;
    int token = -1;
};

struct BackwardGate {
    bool pass = false;
    double norm = 0.0;
    double normRel = 0.0;
    ProbeError gradient;
};

struct AdamGate {
    bool pass = false;
    ProbeError parameter;
};

struct CheckpointGate {
    bool pass = false;
    std::string path;
    size_t bytes = 0;
    int probes = 0;
    double maxAbs = 0.0;
};

struct BenchmarkGate {
    bool pass = false;
    int warmupSteps = 1;
    int timedSteps = BENCH_STEPS;
    double seconds = 0.0;
    double tokensPerSecond = 0.0;
    double cpuTokensPerSecond = 0.0;
    double ratio = 0.0;
    double finalLoss = 0.0;
    bool useful = false;
    bool canonical = false;
};

struct StageProfile {
    bool pass = false;
    int warmupSteps = 1;
    int profiledSteps = 3;
    double forwardSeconds = 0.0;
    double backwardSeconds = 0.0;
    double gradNormSeconds = 0.0;
    double adamwSeconds = 0.0;
    double totalSeconds = 0.0;
};

struct StateProbe {
    std::string slot;
    int index = 0;
    float parameter = 0.0f;
    float moment1 = 0.0f;
    float moment2 = 0.0f;
};


struct PilotDataFile {
    std::string relativePath;
    std::vector<uint16_t> tokens;
    int fullWindows = 0;
};

struct PilotPackageData {
    std::vector<double> lrCandidates;
    std::vector<int> trainIndices;
    std::vector<int> v3EvalIndices;
    std::vector<int> v1EvalIndices;
    int trainSteps = 0;
    int warmupSteps = 0;
    PilotDataFile train;
    PilotDataFile v3Validation;
    PilotDataFile v1Validation;
};

std::string safePilotPath(
    const std::string& root, const std::string& relative) {
    req(!relative.empty() && relative[0] != '/' &&
            relative.find("..") == std::string::npos,
        "unsafe pilot relative path");
    return root + "/" + relative;
}

std::vector<uint16_t> readU16LeFile(const std::string& path) {
    const auto bytes = readBinaryFile(path);
    req(!bytes.empty() && bytes.size() % 2 == 0,
        "pilot uint16 file has invalid byte length: " + path);
    std::vector<uint16_t> result(bytes.size() / 2);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<uint16_t>(
            static_cast<uint16_t>(bytes[i * 2]) |
            (static_cast<uint16_t>(bytes[i * 2 + 1]) << 8));
        req(result[i] < V, "pilot token out of vocabulary");
    }
    return result;
}

std::vector<int> readIndexArray(
    const rapidjson::Value& parent, const char* key) {
    req(parent.HasMember(key) && parent[key].IsArray(),
        std::string("pilot manifest missing index array: ") + key);
    std::vector<int> result;
    for (const auto& value : parent[key].GetArray()) {
        req(value.IsInt() && value.GetInt() >= 0,
            std::string("invalid pilot index: ") + key);
        result.push_back(value.GetInt());
    }
    req(!result.empty(), std::string("empty pilot index array: ") + key);
    return result;
}

PilotDataFile loadPilotDataFile(
    const std::string& root, const rapidjson::Value& data,
    const char* key) {
    req(data.HasMember(key) && data[key].IsObject(),
        std::string("pilot data manifest missing: ") + key);
    const auto& spec = data[key];
    req(spec.HasMember("path") && spec["path"].IsString() &&
            spec.HasMember("full_windows") && spec["full_windows"].IsInt() &&
            spec.HasMember("uint16_tokens") && spec["uint16_tokens"].IsUint64(),
        std::string("invalid pilot data spec: ") + key);
    PilotDataFile file;
    file.relativePath = spec["path"].GetString();
    file.fullWindows = spec["full_windows"].GetInt();
    req(file.fullWindows > 0, "pilot data has zero windows");
    file.tokens = readU16LeFile(safePilotPath(root, file.relativePath));
    req(file.tokens.size() == spec["uint16_tokens"].GetUint64(),
        "pilot data token count mismatch");
    req(static_cast<int>((file.tokens.size() - 1) / S) == file.fullWindows,
        "pilot data full-window count mismatch");
    return file;
}

PilotPackageData loadPilotPackage(const std::string& root) {
    const std::string json = readTextFile(root + "/manifest.json");
    rapidjson::Document doc;
    doc.Parse(json.c_str(), json.size());
    req(!doc.HasParseError() && doc.IsObject(),
        "invalid pilot manifest.json");
    req(doc.HasMember("schema") && doc["schema"].IsString() &&
            std::string(doc["schema"].GetString()) ==
                "model0001_v3_lr_pilot_v1",
        "unsupported pilot schema");
    req(doc.HasMember("source_model_state_sha256") &&
            doc["source_model_state_sha256"].IsString() &&
            std::string(doc["source_model_state_sha256"].GetString()) ==
                "047b0f6ec18046c7a5ae7da707e91a03e26a6819cfec254f8ad541c8ddbf696d",
        "pilot source model SHA mismatch");
    req(doc.HasMember("hard_guards") && doc["hard_guards"].IsObject(),
        "pilot hard guards missing");
    const auto& guards = doc["hard_guards"];
    req(guards.HasMember("test_split_packaged") &&
            guards["test_split_packaged"].IsBool() &&
            !guards["test_split_packaged"].GetBool(),
        "pilot package must not contain test split");
    req(guards.HasMember("dataset_v2_train_bin_packaged") &&
            guards["dataset_v2_train_bin_packaged"].IsBool() &&
            !guards["dataset_v2_train_bin_packaged"].GetBool(),
        "pilot package must not contain Dataset-v2 train bin");
    req(doc.HasMember("protocol") && doc["protocol"].IsObject(),
        "pilot protocol missing");
    const auto& protocol = doc["protocol"];
    req(protocol.HasMember("train_steps_per_candidate") &&
            protocol["train_steps_per_candidate"].IsInt() &&
            protocol.HasMember("warmup_steps") &&
            protocol["warmup_steps"].IsInt() &&
            protocol.HasMember("lr_candidates") &&
            protocol["lr_candidates"].IsArray(),
        "pilot protocol malformed");

    PilotPackageData result;
    result.trainSteps = protocol["train_steps_per_candidate"].GetInt();
    result.warmupSteps = protocol["warmup_steps"].GetInt();
    req(result.trainSteps == 96 && result.warmupSteps == 3,
        "pilot step/warmup contract drift");
    for (const auto& value : protocol["lr_candidates"].GetArray()) {
        req(value.IsNumber(), "pilot LR candidate is not numeric");
        const double lr = value.GetDouble();
        req(std::isfinite(lr) && lr >= 5.0e-5 && lr <= 2.0e-4,
            "pilot LR candidate outside locked CPT pilot range");
        result.lrCandidates.push_back(lr);
    }
    req(result.lrCandidates.size() == 3,
        "pilot must contain exactly three LR candidates");

    req(doc.HasMember("indices") && doc["indices"].IsObject(),
        "pilot indices missing");
    const auto& indices = doc["indices"];
    result.trainIndices = readIndexArray(indices, "train");
    result.v3EvalIndices = readIndexArray(indices, "v3_validation");
    result.v1EvalIndices = readIndexArray(indices, "v1_validation");
    req(static_cast<int>(result.trainIndices.size()) == result.trainSteps,
        "pilot train index count mismatch");
    req(result.v3EvalIndices.size() == 24 &&
            result.v1EvalIndices.size() == 24,
        "pilot eval-window count drift");

    req(doc.HasMember("data") && doc["data"].IsObject(),
        "pilot data section missing");
    const auto& data = doc["data"];
    result.train = loadPilotDataFile(root, data, "v3_train");
    result.v3Validation =
        loadPilotDataFile(root, data, "v3_validation");
    result.v1Validation =
        loadPilotDataFile(root, data, "v1_validation");

    auto validateIndices = [](const std::vector<int>& values, int limit,
                              const char* label) {
        for (int value : values) {
            req(value >= 0 && value < limit,
                std::string("pilot index outside ") + label);
        }
    };
    validateIndices(result.trainIndices, result.train.fullWindows, "v3 train");
    validateIndices(
        result.v3EvalIndices, result.v3Validation.fullWindows, "v3 validation");
    validateIndices(
        result.v1EvalIndices, result.v1Validation.fullWindows, "v1 validation");
    return result;
}

class NativeTrainer {
public:
    NativeTrainer(const Bundle& bundle, std::string workDirectory)
        : bundle_(bundle),
          workDirectory_(std::move(workDirectory)),
          runtime_(processRuntime()) {
        stagePath_ = workDirectory_ + "/last_native_stage.txt";
    }

    NativeGateResult run(const std::function<double()>& cpuBaseline);
    NativePilotResult runPilot(const PilotPackageData& pilot);
    const std::string& currentStage() const { return currentStage_; }

private:
    const Bundle& bundle_;
    std::string workDirectory_;
    std::string stagePath_;
    std::string currentStage_ = "native:initialize";
    ProcessRuntime& runtime_;
    std::map<std::string, SlotBuffers> slots_;
    std::array<LayerBuffers, LAYERS> layers_{};
    SlotBuffers* embedding_ = nullptr;
    SlotBuffers* finalNorm_ = nullptr;
    uint64_t optimizerStep_ = 0;
    float currentLearningRate_ = 0.0f;
    size_t persistentBytes_ = 0;
    size_t activationBytes_ = 0;
    size_t workspaceBytes_ = 0;

    cl_mem tokens_ = nullptr;
    cl_mem targets_ = nullptr;
    cl_mem ropeCos_ = nullptr;
    cl_mem ropeSin_ = nullptr;
    cl_mem uniqueTokenIds_ = nullptr;
    cl_mem uniquePositions_ = nullptr;
    cl_mem uniqueOffsets_ = nullptr;
    int uniqueTokenCount_ = 0;

    std::array<cl_mem, LAYERS> layerInput_{};
    std::array<cl_mem, LAYERS> layerAttn_{};

    // Retain the backward-critical forward activations. The previous gate
    // recomputed a complete attention+FFN subgraph inside every backward layer;
    // profiling proved that recompute alone consumed a material fraction of
    // backward time. Model #0001 has ample memory headroom, so trade ~54 MiB
    // for eliminating that repeated compute.
    std::array<cl_mem, LAYERS> retainedQRoPe_{};
    std::array<cl_mem, LAYERS> retainedKRepeat_{};
    std::array<cl_mem, LAYERS> retainedVRepeat_{};
    std::array<cl_mem, LAYERS> retainedProbability_{};
    std::array<cl_mem, LAYERS> retainedAttentionFlat_{};
    std::array<cl_mem, LAYERS> retainedFfnNorm_{};
    std::array<cl_mem, LAYERS> retainedGate_{};
    std::array<cl_mem, LAYERS> retainedUp_{};
    std::array<cl_mem, LAYERS> retainedFf_{};

    cl_mem modelOutput_ = nullptr;
    cl_mem finalNormalized_ = nullptr;
    cl_mem logits_ = nullptr;
    cl_mem dLogits_ = nullptr;
    cl_mem rowLoss_ = nullptr;
    cl_mem loss_ = nullptr;

    cl_mem norm_ = nullptr;
    cl_mem q_ = nullptr;
    cl_mem k_ = nullptr;
    cl_mem v_ = nullptr;
    cl_mem qRope_ = nullptr;
    cl_mem kRope_ = nullptr;
    cl_mem kRepeat_ = nullptr;
    cl_mem vRepeat_ = nullptr;
    cl_mem scores_ = nullptr;
    cl_mem probability_ = nullptr;
    cl_mem context_ = nullptr;
    cl_mem attentionFlat_ = nullptr;
    cl_mem gate_ = nullptr;
    cl_mem up_ = nullptr;
    cl_mem ff_ = nullptr;

    cl_mem gradA_ = nullptr;
    cl_mem gradB_ = nullptr;
    cl_mem tempD_ = nullptr;
    cl_mem dFf_ = nullptr;
    cl_mem tempFf_ = nullptr;
    cl_mem dContext_ = nullptr;
    cl_mem dProbability_ = nullptr;
    cl_mem dQ_ = nullptr;
    cl_mem dKRepeat_ = nullptr;
    cl_mem dVRepeat_ = nullptr;
    cl_mem dK_ = nullptr;
    cl_mem dV_ = nullptr;
    cl_mem normPartials_ = nullptr;
    cl_mem globalNorm_ = nullptr;
    cl_mem rmsInvRows_ = nullptr;
    cl_mem probeIndices_ = nullptr;
    cl_mem probeOutput_ = nullptr;

    void mark(const std::string& value) {
        currentStage_ = value;
        try {
            atomicWrite(stagePath_, value.data(), value.size());
        } catch (...) {
            // A diagnostic marker must never change numerical behavior.
        }
    }

    cl_mem allocateElements(size_t count, size_t* category) {
        const size_t bytes = bytesFor(count);
        cl_mem result = runtime_.allocate(bytes);
        if (category) *category += bytes;
        return result;
    }

    SlotBuffers& slot(const std::string& name) {
        auto it = slots_.find(name);
        req(it != slots_.end(), "missing native slot " + name);
        return it->second;
    }

    void initializeSlots();
    void resetToSourceState();
    void initializeInputs();
    void setTrainingWindow(const int32_t* tokens257);
    void setLearningRate(float lr);
    void initializeActivations();
    void wireLayers();
    void validateGeometry() const;
    ProbeError validateWeightLoad();

    void linearForward(cl_mem a, cl_mem w, cl_mem c, int m, int k, int n);
    void linearDInput(cl_mem dy, cl_mem w, cl_mem dx, int m, int k, int n);
    void linearDWeight(cl_mem dy, cl_mem a, cl_mem dw, int m, int k, int n);
    void rmsForward(cl_mem x, SlotBuffers& weight, cl_mem y);
    void rmsBackward(
        cl_mem x, SlotBuffers& weight, cl_mem dy, cl_mem dx);
    void splitHeads(cl_mem flat, cl_mem heads, int count);
    void mergeHeads(cl_mem heads, cl_mem flat, int count);
    void ropeForward(cl_mem x, cl_mem y, int heads);
    void ropeBackward(cl_mem dy, cl_mem dx, int heads);
    void repeatKv(cl_mem source, cl_mem destination);
    void reduceGqa(cl_mem source, cl_mem destination);
    void bmmNt(
        cl_mem a, cl_mem b, cl_mem c,
        int batches, int m, int n, int k, float scale);
    void bmmNn(
        cl_mem a, cl_mem b, cl_mem c,
        int batches, int m, int n, int k, float scale);
    void bmmLeftT(
        cl_mem a, cl_mem b, cl_mem c,
        int batches, int m, int n, int k, float scale);
    void add(cl_mem a, cl_mem b, cl_mem out, int count);
    void addInPlace(cl_mem destination, cl_mem source, int count);
    void recomputeLayer(int layer);
    void forward();
    void backward();
    void computeGlobalNorm();
    void adamStep();
    void fullTrainingStep();
    float readLoss();
    float readGlobalNorm();
    void setWindowFromU16(const PilotDataFile& data, int windowIndex);
    double evaluateCe(
        const PilotDataFile& data, const std::vector<int>& indices);

    std::vector<float> gather(cl_mem source, const std::vector<int>& indices);
    ForwardGate checkForward();
    BackwardGate checkBackward();
    AdamGate checkAdam();
    std::vector<StateProbe> captureStateProbes();
    CheckpointGate checkpointRoundTrip();
    void saveCheckpoint(const std::string& path, size_t* bytesWritten);
    void loadCheckpoint(const std::string& path);
    BenchmarkGate benchmark(const std::function<double()>& cpuBaseline);
    StageProfile profileStages();
    std::string profileKernelsJson();
    std::string memoryJson() const;
};

void NativeTrainer::validateGeometry() const {
    req(bundle_.tensors.size() == 74, "native trainer requires exactly 74 tensors");
    req(bundle_.parameterCount == 19145088, "native trainer parameter count drift");
    req(bundle_.config.seqLen == S && bundle_.config.vocabSize == V &&
        bundle_.config.dModel == D && bundle_.config.nHeads == HQ &&
        bundle_.config.nKvHeads == HKV && bundle_.config.headDim == HD &&
        bundle_.config.dFf == FF && bundle_.config.nLayers == LAYERS,
        "native trainer geometry mismatch");
    req(bundle_.ropeStyle == "interleaved",
        "native trainer requires declared interleaved RoPE");
    req(std::abs(bundle_.adam.beta1 - 0.9) <= 1.0e-12 &&
        std::abs(bundle_.adam.beta2 - 0.95) <= 1.0e-12 &&
        std::abs(bundle_.adam.eps - 1.0e-8) <= 1.0e-16 &&
        std::abs(bundle_.adam.gateLr - 1.0e-4) <= 1.0e-12,
        "native trainer frozen AdamW hyperparameters mismatch");
}

void NativeTrainer::initializeSlots() {
    validateGeometry();
    mark("native:gate1:weight_load:start");
    size_t elementTotal = 0;
    for (const auto& pair : bundle_.tensors) {
        SlotBuffers buffers;
        buffers.name = pair.first;
        buffers.shape = pair.second.shape;
        buffers.elements = pair.second.data.size();
        buffers.weightDecay = static_cast<float>(
            bundle_.adam.slotWeightDecay.at(pair.first));
        const size_t bytes = bytesFor(buffers.elements);
        buffers.parameter = runtime_.allocate(bytes);
        buffers.gradient = runtime_.allocate(bytes);
        buffers.moment1 = runtime_.allocate(bytes);
        buffers.moment2 = runtime_.allocate(bytes);
        persistentBytes_ += bytes * 4;
        runtime_.write(buffers.parameter, pair.second.data.data(), bytes);
        runtime_.zero(buffers.gradient, bytes);
        runtime_.zero(buffers.moment1, bytes);
        runtime_.zero(buffers.moment2, bytes);
        elementTotal += buffers.elements;
        slots_.emplace(pair.first, std::move(buffers));
    }
    req(elementTotal == static_cast<size_t>(bundle_.parameterCount),
        "native trainer loaded parameter count mismatch");
    runtime_.finish();
    wireLayers();
}

void NativeTrainer::resetToSourceState() {
    // Correctness gates intentionally mutate the native state (AdamW parity
    // performs one real optimizer update and checkpoint verification reloads
    // that updated state).  Sustained performance must not inherit that
    // mutation: the CPU denominator starts from the immutable CPT-v2 source
    // weights with fresh zero moments.  Restore the exact same state here.
    for (auto& pair : slots_) {
        auto& value = pair.second;
        const auto& source = bundle_.tensor(pair.first).data;
        req(source.size() == value.elements,
            "source tensor size drift during benchmark reset: " + value.name);
        const size_t bytes = bytesFor(value.elements);
        runtime_.write(value.parameter, source.data(), bytes);
        runtime_.zero(value.gradient, bytes);
        runtime_.zero(value.moment1, bytes);
        runtime_.zero(value.moment2, bytes);
    }
    optimizerStep_ = 0;
    setTrainingWindow(bundle_.sampleTokens.data());
    setLearningRate(static_cast<float>(bundle_.adam.gateLr));
    runtime_.finish();
}

void NativeTrainer::wireLayers() {
    embedding_ = &slot("tok_embeddings.weight");
    finalNorm_ = &slot("final_norm.weight");
    for (int layer = 0; layer < LAYERS; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        auto& item = layers_[layer];
        item.attnNorm = &slot(prefix + "attn_norm.weight");
        item.q = &slot(prefix + "q_proj.weight");
        item.k = &slot(prefix + "k_proj.weight");
        item.v = &slot(prefix + "v_proj.weight");
        item.o = &slot(prefix + "o_proj.weight");
        item.ffnNorm = &slot(prefix + "ffn_norm.weight");
        item.gate = &slot(prefix + "gate_proj.weight");
        item.up = &slot(prefix + "up_proj.weight");
        item.down = &slot(prefix + "down_proj.weight");
    }
    auto expectShape = [](const SlotBuffers& value,
                          std::initializer_list<int> expected) {
        req(value.shape == std::vector<int>(expected),
            "native tensor shape mismatch: " + value.name);
    };
    expectShape(*embedding_, {V, D});
    expectShape(*finalNorm_, {D});
    for (const auto& item : layers_) {
        expectShape(*item.attnNorm, {D});
        expectShape(*item.q, {D, D});
        expectShape(*item.k, {HKV * HD, D});
        expectShape(*item.v, {HKV * HD, D});
        expectShape(*item.o, {D, D});
        expectShape(*item.ffnNorm, {D});
        expectShape(*item.gate, {FF, D});
        expectShape(*item.up, {FF, D});
        expectShape(*item.down, {D, FF});
    }
}

void NativeTrainer::initializeInputs() {
    tokens_ = runtime_.allocate(S * sizeof(int32_t));
    targets_ = runtime_.allocate(S * sizeof(int32_t));

    std::vector<float> cosines(S * HD), sines(S * HD);
    for (int position = 0; position < S; ++position) {
        for (int pair = 0; pair < HD / 2; ++pair) {
            const double inverse = std::pow(
                bundle_.config.ropeTheta, -2.0 * pair / HD);
            const float cosine = static_cast<float>(
                std::cos(position * inverse));
            const float sine = static_cast<float>(
                std::sin(position * inverse));
            cosines[position * HD + pair * 2] = cosine;
            cosines[position * HD + pair * 2 + 1] = cosine;
            sines[position * HD + pair * 2] = sine;
            sines[position * HD + pair * 2 + 1] = sine;
        }
    }
    ropeCos_ = allocateElements(S * HD, &persistentBytes_);
    ropeSin_ = allocateElements(S * HD, &persistentBytes_);
    runtime_.write(ropeCos_, cosines.data(), bytesFor(cosines.size()));
    runtime_.write(ropeSin_, sines.data(), bytesFor(sines.size()));

    // Production windows can have a different number of unique tokens than
    // the parity sample. Allocate worst-case tables once and only rewrite their
    // contents per step.
    uniqueTokenIds_ = runtime_.allocate(S * sizeof(int32_t));
    uniquePositions_ = runtime_.allocate(S * sizeof(int32_t));
    uniqueOffsets_ = runtime_.allocate((S + 1) * sizeof(int32_t));
    setTrainingWindow(bundle_.sampleTokens.data());
    setLearningRate(static_cast<float>(bundle_.adam.gateLr));
}

void NativeTrainer::setTrainingWindow(const int32_t* tokens257) {
    req(tokens257 != nullptr, "null training window");
    for (int i = 0; i <= S; ++i) {
        req(tokens257[i] >= 0 && tokens257[i] < V,
            "training window token out of vocabulary");
    }
    runtime_.write(tokens_, tokens257, S * sizeof(int32_t));
    runtime_.write(targets_, tokens257 + 1, S * sizeof(int32_t));

    std::map<int32_t, std::vector<int32_t>> byToken;
    for (int position = 0; position < S; ++position) {
        byToken[tokens257[position]].push_back(position);
    }
    std::vector<int32_t> ids;
    std::vector<int32_t> positions;
    std::vector<int32_t> offsets;
    offsets.reserve(byToken.size() + 1);
    offsets.push_back(0);
    for (const auto& item : byToken) {
        ids.push_back(item.first);
        positions.insert(positions.end(), item.second.begin(), item.second.end());
        offsets.push_back(static_cast<int32_t>(positions.size()));
    }
    req(!ids.empty() && ids.size() <= S && positions.size() == S &&
        offsets.size() == ids.size() + 1,
        "embedding reduction position table invalid");
    uniqueTokenCount_ = static_cast<int>(ids.size());
    runtime_.write(uniqueTokenIds_, ids.data(), ids.size() * sizeof(int32_t));
    runtime_.write(
        uniquePositions_, positions.data(), positions.size() * sizeof(int32_t));
    runtime_.write(
        uniqueOffsets_, offsets.data(), offsets.size() * sizeof(int32_t));
}

void NativeTrainer::setLearningRate(float lr) {
    req(std::isfinite(lr) && lr > 0.0f, "production learning rate invalid");
    currentLearningRate_ = lr;
}

void NativeTrainer::initializeActivations() {
    constexpr size_t sd = static_cast<size_t>(S) * D;
    constexpr size_t skv = static_cast<size_t>(S) * HKV * HD;
    constexpr size_t sh = static_cast<size_t>(S) * HQ * HD;
    constexpr size_t attention = static_cast<size_t>(HQ) * S * S;
    constexpr size_t sff = static_cast<size_t>(S) * FF;
    for (int layer = 0; layer < LAYERS; ++layer) {
        layerInput_[layer] = allocateElements(sd, &activationBytes_);
        layerAttn_[layer] = allocateElements(sd, &activationBytes_);
        retainedQRoPe_[layer] = allocateElements(sh, &activationBytes_);
        retainedKRepeat_[layer] = allocateElements(sh, &activationBytes_);
        retainedVRepeat_[layer] = allocateElements(sh, &activationBytes_);
        retainedProbability_[layer] =
            allocateElements(attention, &activationBytes_);
        retainedAttentionFlat_[layer] =
            allocateElements(sd, &activationBytes_);
        retainedFfnNorm_[layer] = allocateElements(sd, &activationBytes_);
        retainedGate_[layer] = allocateElements(sff, &activationBytes_);
        retainedUp_[layer] = allocateElements(sff, &activationBytes_);
        retainedFf_[layer] = allocateElements(sff, &activationBytes_);
    }
    modelOutput_ = allocateElements(sd, &activationBytes_);
    finalNormalized_ = allocateElements(sd, &activationBytes_);
    logits_ = allocateElements(static_cast<size_t>(S) * V, &activationBytes_);
    dLogits_ = allocateElements(static_cast<size_t>(S) * V, &activationBytes_);
    rowLoss_ = allocateElements(S, &activationBytes_);
    loss_ = allocateElements(1, &activationBytes_);

    norm_ = allocateElements(sd, &workspaceBytes_);
    q_ = allocateElements(sh, &workspaceBytes_);
    k_ = allocateElements(skv, &workspaceBytes_);
    v_ = allocateElements(skv, &workspaceBytes_);
    qRope_ = allocateElements(sh, &workspaceBytes_);
    kRope_ = allocateElements(skv, &workspaceBytes_);
    kRepeat_ = allocateElements(sh, &workspaceBytes_);
    vRepeat_ = allocateElements(sh, &workspaceBytes_);
    scores_ = allocateElements(attention, &workspaceBytes_);
    probability_ = allocateElements(attention, &workspaceBytes_);
    context_ = allocateElements(sh, &workspaceBytes_);
    attentionFlat_ = allocateElements(sd, &workspaceBytes_);
    gate_ = allocateElements(sff, &workspaceBytes_);
    up_ = allocateElements(sff, &workspaceBytes_);
    ff_ = allocateElements(sff, &workspaceBytes_);

    gradA_ = allocateElements(sd, &workspaceBytes_);
    gradB_ = allocateElements(sd, &workspaceBytes_);
    tempD_ = allocateElements(sd, &workspaceBytes_);
    dFf_ = allocateElements(sff, &workspaceBytes_);
    tempFf_ = allocateElements(sff, &workspaceBytes_);
    dContext_ = allocateElements(sh, &workspaceBytes_);
    dProbability_ = allocateElements(attention, &workspaceBytes_);
    dQ_ = allocateElements(sh, &workspaceBytes_);
    dKRepeat_ = allocateElements(sh, &workspaceBytes_);
    dVRepeat_ = allocateElements(sh, &workspaceBytes_);
    dK_ = allocateElements(skv, &workspaceBytes_);
    dV_ = allocateElements(skv, &workspaceBytes_);
    normPartials_ = allocateElements(
        slots_.size() * REDUCE_GROUPS, &workspaceBytes_);
    globalNorm_ = allocateElements(1, &workspaceBytes_);
    rmsInvRows_ = allocateElements(S, &workspaceBytes_);
    probeIndices_ = runtime_.allocate(4096 * sizeof(int32_t));
    probeOutput_ = allocateElements(4096, &workspaceBytes_);
}

void NativeTrainer::linearForward(
    cl_mem a, cl_mem w, cl_mem c, int m, int k, int n) {
    cl_kernel kernel = runtime_.kernel("linear_forward");
    runtime_.argument(kernel, 0, a);
    runtime_.argument(kernel, 1, w);
    runtime_.argument(kernel, 2, c);
    runtime_.argument(kernel, 3, m);
    runtime_.argument(kernel, 4, k);
    runtime_.argument(kernel, 5, n);
    runtime_.enqueue2Micro2x2(kernel, n, m);
}

void NativeTrainer::linearDInput(
    cl_mem dy, cl_mem w, cl_mem dx, int m, int k, int n) {
    cl_kernel kernel = runtime_.kernel("linear_dinput");
    runtime_.argument(kernel, 0, dy);
    runtime_.argument(kernel, 1, w);
    runtime_.argument(kernel, 2, dx);
    runtime_.argument(kernel, 3, m);
    runtime_.argument(kernel, 4, k);
    runtime_.argument(kernel, 5, n);
    runtime_.enqueue2Micro2x2(kernel, k, m);
}

void NativeTrainer::linearDWeight(
    cl_mem dy, cl_mem a, cl_mem dw, int m, int k, int n) {
    cl_kernel kernel = runtime_.kernel("linear_dweight");
    runtime_.argument(kernel, 0, dy);
    runtime_.argument(kernel, 1, a);
    runtime_.argument(kernel, 2, dw);
    runtime_.argument(kernel, 3, m);
    runtime_.argument(kernel, 4, k);
    runtime_.argument(kernel, 5, n);
    runtime_.enqueue2Micro2x2(kernel, k, n);
}

void NativeTrainer::rmsForward(
    cl_mem x, SlotBuffers& weight, cl_mem y) {
    const int rows = S, d = D;
    const float eps = static_cast<float>(bundle_.config.rmsNormEps);
    cl_kernel kernel = runtime_.kernel("rmsnorm_forward");
    runtime_.argument(kernel, 0, x);
    runtime_.argument(kernel, 1, weight.parameter);
    runtime_.argument(kernel, 2, y);
    runtime_.argument(kernel, 3, rows);
    runtime_.argument(kernel, 4, d);
    runtime_.argument(kernel, 5, eps);
    runtime_.enqueue1(kernel, rows);
}

void NativeTrainer::rmsBackward(
    cl_mem x, SlotBuffers& weight, cl_mem dy, cl_mem dx) {
    const int rows = S, d = D;
    const float eps = static_cast<float>(bundle_.config.rmsNormEps);
    cl_kernel kdx = runtime_.kernel("rmsnorm_dx");
    runtime_.argument(kdx, 0, x);
    runtime_.argument(kdx, 1, weight.parameter);
    runtime_.argument(kdx, 2, dy);
    runtime_.argument(kdx, 3, dx);
    runtime_.argument(kdx, 4, rmsInvRows_);
    runtime_.argument(kdx, 5, rows);
    runtime_.argument(kdx, 6, d);
    runtime_.argument(kdx, 7, eps);
    runtime_.enqueue1(kdx, rows);

    cl_kernel kdw = runtime_.kernel("rmsnorm_dw");
    runtime_.argument(kdw, 0, x);
    runtime_.argument(kdw, 1, dy);
    runtime_.argument(kdw, 2, rmsInvRows_);
    runtime_.argument(kdw, 3, weight.gradient);
    runtime_.argument(kdw, 4, rows);
    runtime_.argument(kdw, 5, d);
    runtime_.enqueue1(kdw, d);
}

void NativeTrainer::splitHeads(cl_mem flat, cl_mem headsBuffer, int heads) {
    const int rows = S, hd = HD;
    cl_kernel kernel = runtime_.kernel("split_heads");
    runtime_.argument(kernel, 0, flat);
    runtime_.argument(kernel, 1, headsBuffer);
    runtime_.argument(kernel, 2, rows);
    runtime_.argument(kernel, 3, heads);
    runtime_.argument(kernel, 4, hd);
    runtime_.enqueue1(kernel, static_cast<size_t>(rows) * heads * hd);
}

void NativeTrainer::mergeHeads(cl_mem headsBuffer, cl_mem flat, int heads) {
    const int rows = S, hd = HD;
    cl_kernel kernel = runtime_.kernel("merge_heads");
    runtime_.argument(kernel, 0, headsBuffer);
    runtime_.argument(kernel, 1, flat);
    runtime_.argument(kernel, 2, rows);
    runtime_.argument(kernel, 3, heads);
    runtime_.argument(kernel, 4, hd);
    runtime_.enqueue1(kernel, static_cast<size_t>(rows) * heads * hd);
}

void NativeTrainer::ropeForward(cl_mem x, cl_mem y, int heads) {
    const int rows = S, hd = HD;
    cl_kernel kernel = runtime_.kernel("rope_forward");
    runtime_.argument(kernel, 0, x);
    runtime_.argument(kernel, 1, ropeCos_);
    runtime_.argument(kernel, 2, ropeSin_);
    runtime_.argument(kernel, 3, y);
    runtime_.argument(kernel, 4, heads);
    runtime_.argument(kernel, 5, rows);
    runtime_.argument(kernel, 6, hd);
    runtime_.enqueue1(
        kernel, static_cast<size_t>(heads) * rows * (hd / 2));
}

void NativeTrainer::ropeBackward(cl_mem dy, cl_mem dx, int heads) {
    const int rows = S, hd = HD;
    cl_kernel kernel = runtime_.kernel("rope_backward");
    runtime_.argument(kernel, 0, dy);
    runtime_.argument(kernel, 1, ropeCos_);
    runtime_.argument(kernel, 2, ropeSin_);
    runtime_.argument(kernel, 3, dx);
    runtime_.argument(kernel, 4, heads);
    runtime_.argument(kernel, 5, rows);
    runtime_.argument(kernel, 6, hd);
    runtime_.enqueue1(
        kernel, static_cast<size_t>(heads) * rows * (hd / 2));
}

void NativeTrainer::repeatKv(cl_mem source, cl_mem destination) {
    const int hq = HQ, hkv = HKV, rows = S, hd = HD;
    cl_kernel kernel = runtime_.kernel("repeat_kv");
    runtime_.argument(kernel, 0, source);
    runtime_.argument(kernel, 1, destination);
    runtime_.argument(kernel, 2, hq);
    runtime_.argument(kernel, 3, hkv);
    runtime_.argument(kernel, 4, rows);
    runtime_.argument(kernel, 5, hd);
    runtime_.enqueue1(kernel, static_cast<size_t>(hq) * rows * hd);
}

void NativeTrainer::reduceGqa(cl_mem source, cl_mem destination) {
    const int hq = HQ, hkv = HKV, rows = S, hd = HD;
    cl_kernel kernel = runtime_.kernel("reduce_gqa");
    runtime_.argument(kernel, 0, source);
    runtime_.argument(kernel, 1, destination);
    runtime_.argument(kernel, 2, hq);
    runtime_.argument(kernel, 3, hkv);
    runtime_.argument(kernel, 4, rows);
    runtime_.argument(kernel, 5, hd);
    runtime_.enqueue1(kernel, static_cast<size_t>(hkv) * rows * hd);
}

void NativeTrainer::bmmNt(
    cl_mem a, cl_mem b, cl_mem c,
    int batches, int m, int n, int k, float scale) {
    cl_kernel kernel = runtime_.kernel("bmm_nt");
    runtime_.argument(kernel, 0, a);
    runtime_.argument(kernel, 1, b);
    runtime_.argument(kernel, 2, c);
    runtime_.argument(kernel, 3, batches);
    runtime_.argument(kernel, 4, m);
    runtime_.argument(kernel, 5, n);
    runtime_.argument(kernel, 6, k);
    runtime_.argument(kernel, 7, scale);
    runtime_.enqueue2(kernel, n, static_cast<size_t>(batches) * m);
}

void NativeTrainer::bmmNn(
    cl_mem a, cl_mem b, cl_mem c,
    int batches, int m, int n, int k, float scale) {
    cl_kernel kernel = runtime_.kernel("bmm_nn");
    runtime_.argument(kernel, 0, a);
    runtime_.argument(kernel, 1, b);
    runtime_.argument(kernel, 2, c);
    runtime_.argument(kernel, 3, batches);
    runtime_.argument(kernel, 4, m);
    runtime_.argument(kernel, 5, n);
    runtime_.argument(kernel, 6, k);
    runtime_.argument(kernel, 7, scale);
    runtime_.enqueue2(kernel, k, static_cast<size_t>(batches) * m);
}

void NativeTrainer::bmmLeftT(
    cl_mem a, cl_mem b, cl_mem c,
    int batches, int m, int n, int k, float scale) {
    cl_kernel kernel = runtime_.kernel("bmm_left_t");
    runtime_.argument(kernel, 0, a);
    runtime_.argument(kernel, 1, b);
    runtime_.argument(kernel, 2, c);
    runtime_.argument(kernel, 3, batches);
    runtime_.argument(kernel, 4, m);
    runtime_.argument(kernel, 5, n);
    runtime_.argument(kernel, 6, k);
    runtime_.argument(kernel, 7, scale);
    runtime_.enqueue2(kernel, k, static_cast<size_t>(batches) * n);
}

void NativeTrainer::add(cl_mem a, cl_mem b, cl_mem out, int count) {
    cl_kernel kernel = runtime_.kernel("vector_add");
    runtime_.argument(kernel, 0, a);
    runtime_.argument(kernel, 1, b);
    runtime_.argument(kernel, 2, out);
    runtime_.argument(kernel, 3, count);
    runtime_.enqueue1(kernel, count);
}

void NativeTrainer::addInPlace(
    cl_mem destination, cl_mem source, int count) {
    cl_kernel kernel = runtime_.kernel("vector_add_inplace");
    runtime_.argument(kernel, 0, destination);
    runtime_.argument(kernel, 1, source);
    runtime_.argument(kernel, 2, count);
    runtime_.enqueue1(kernel, count);
}

void NativeTrainer::recomputeLayer(int layerIndex) {
    auto& layer = layers_[layerIndex];
    cl_mem input = layerInput_[layerIndex];
    rmsForward(input, *layer.attnNorm, norm_);
    linearForward(norm_, layer.q->parameter, attentionFlat_, S, D, D);
    splitHeads(attentionFlat_, q_, HQ);
    linearForward(norm_, layer.k->parameter, attentionFlat_, S, D, HKV * HD);
    splitHeads(attentionFlat_, k_, HKV);
    linearForward(norm_, layer.v->parameter, attentionFlat_, S, D, HKV * HD);
    splitHeads(attentionFlat_, v_, HKV);
    ropeForward(q_, qRope_, HQ);
    ropeForward(k_, kRope_, HKV);
    repeatKv(kRope_, kRepeat_);
    repeatKv(v_, vRepeat_);
    bmmNt(qRope_, kRepeat_, scores_, HQ, S, S, HD,
        1.0f / std::sqrt(static_cast<float>(HD)));

    const int rows = HQ * S, seq = S;
    cl_kernel softmax = runtime_.kernel("causal_softmax_forward");
    runtime_.argument(softmax, 0, scores_);
    runtime_.argument(softmax, 1, probability_);
    runtime_.argument(softmax, 2, rows);
    runtime_.argument(softmax, 3, seq);
    runtime_.enqueue1(softmax, rows);
    bmmNn(probability_, vRepeat_, context_, HQ, S, S, HD, 1.0f);
    mergeHeads(context_, attentionFlat_, HQ);

    rmsForward(layerAttn_[layerIndex], *layer.ffnNorm, norm_);
    linearForward(norm_, layer.gate->parameter, gate_, S, D, FF);
    linearForward(norm_, layer.up->parameter, up_, S, D, FF);
    const int count = S * FF;
    cl_kernel silu = runtime_.kernel("silu_multiply");
    runtime_.argument(silu, 0, gate_);
    runtime_.argument(silu, 1, up_);
    runtime_.argument(silu, 2, ff_);
    runtime_.argument(silu, 3, count);
    runtime_.enqueue1(silu, count);
}

void NativeTrainer::forward() {
    const int sd = S * D;
    cl_kernel embedding = runtime_.kernel("embedding_lookup");
    const int rows = S, d = D;
    runtime_.argument(embedding, 0, tokens_);
    runtime_.argument(embedding, 1, embedding_->parameter);
    runtime_.argument(embedding, 2, layerInput_[0]);
    runtime_.argument(embedding, 3, rows);
    runtime_.argument(embedding, 4, d);
    runtime_.enqueue1(embedding, sd);

    for (int layerIndex = 0; layerIndex < LAYERS; ++layerIndex) {
        auto& layer = layers_[layerIndex];
        cl_mem input = layerInput_[layerIndex];
        rmsForward(input, *layer.attnNorm, norm_);
        linearForward(norm_, layer.q->parameter, attentionFlat_, S, D, D);
        splitHeads(attentionFlat_, q_, HQ);
        linearForward(norm_, layer.k->parameter, attentionFlat_, S, D, HKV * HD);
        splitHeads(attentionFlat_, k_, HKV);
        linearForward(norm_, layer.v->parameter, attentionFlat_, S, D, HKV * HD);
        splitHeads(attentionFlat_, v_, HKV);
        ropeForward(q_, retainedQRoPe_[layerIndex], HQ);
        ropeForward(k_, kRope_, HKV);
        repeatKv(kRope_, retainedKRepeat_[layerIndex]);
        repeatKv(v_, retainedVRepeat_[layerIndex]);
        bmmNt(retainedQRoPe_[layerIndex], retainedKRepeat_[layerIndex],
            scores_, HQ, S, S, HD,
            1.0f / std::sqrt(static_cast<float>(HD)));

        const int attentionRows = HQ * S, seq = S;
        cl_kernel softmax = runtime_.kernel("causal_softmax_forward");
        runtime_.argument(softmax, 0, scores_);
        runtime_.argument(softmax, 1, retainedProbability_[layerIndex]);
        runtime_.argument(softmax, 2, attentionRows);
        runtime_.argument(softmax, 3, seq);
        runtime_.enqueue1(softmax, attentionRows);
        bmmNn(retainedProbability_[layerIndex], retainedVRepeat_[layerIndex],
            context_, HQ, S, S, HD, 1.0f);
        mergeHeads(context_, retainedAttentionFlat_[layerIndex], HQ);
        linearForward(retainedAttentionFlat_[layerIndex],
            layer.o->parameter, tempD_, S, D, D);
        add(input, tempD_, layerAttn_[layerIndex], sd);

        rmsForward(layerAttn_[layerIndex], *layer.ffnNorm,
            retainedFfnNorm_[layerIndex]);
        linearForward(retainedFfnNorm_[layerIndex], layer.gate->parameter,
            retainedGate_[layerIndex], S, D, FF);
        linearForward(retainedFfnNorm_[layerIndex], layer.up->parameter,
            retainedUp_[layerIndex], S, D, FF);
        const int ffCount = S * FF;
        cl_kernel silu = runtime_.kernel("silu_multiply");
        runtime_.argument(silu, 0, retainedGate_[layerIndex]);
        runtime_.argument(silu, 1, retainedUp_[layerIndex]);
        runtime_.argument(silu, 2, retainedFf_[layerIndex]);
        runtime_.argument(silu, 3, ffCount);
        runtime_.enqueue1(silu, ffCount);
        linearForward(retainedFf_[layerIndex],
            layer.down->parameter, tempD_, S, FF, D);
        cl_mem output = layerIndex + 1 < LAYERS
            ? layerInput_[layerIndex + 1]
            : modelOutput_;
        add(layerAttn_[layerIndex], tempD_, output, sd);
    }

    rmsForward(modelOutput_, *finalNorm_, finalNormalized_);
    linearForward(
        finalNormalized_, embedding_->parameter, logits_, S, D, V);
    cl_kernel ce = runtime_.kernel("cross_entropy_forward_backward");
    const int vocab = V;
    runtime_.argument(ce, 0, logits_);
    runtime_.argument(ce, 1, targets_);
    runtime_.argument(ce, 2, rowLoss_);
    runtime_.argument(ce, 3, dLogits_);
    runtime_.argument(ce, 4, rows);
    runtime_.argument(ce, 5, vocab);
    runtime_.enqueue1(ce, rows);
    cl_kernel mean = runtime_.kernel("mean_rows");
    runtime_.argument(mean, 0, rowLoss_);
    runtime_.argument(mean, 1, loss_);
    runtime_.argument(mean, 2, rows);
    runtime_.enqueue1(mean, 1, 1);
}

void NativeTrainer::backward() {
    const int sd = S * D;
    const float attentionScale = 1.0f / std::sqrt(static_cast<float>(HD));

    // Tied LM head: this writes the direct output-projection contribution to
    // the shared embedding gradient. The lookup contribution is added after
    // the reverse transformer pass.
    linearDWeight(
        dLogits_, finalNormalized_, embedding_->gradient, S, D, V);
    linearDInput(
        dLogits_, embedding_->parameter, gradA_, S, D, V);
    rmsBackward(modelOutput_, *finalNorm_, gradA_, gradB_);
    cl_mem current = gradB_;
    cl_mem ffnResidualGradient = gradA_;

    for (int layerIndex = LAYERS - 1; layerIndex >= 0; --layerIndex) {
        auto& layer = layers_[layerIndex];

        // FFN residual branch.
        linearDWeight(current, retainedFf_[layerIndex],
            layer.down->gradient, S, FF, D);
        linearDInput(current, layer.down->parameter, dFf_, S, FF, D);

        const int ffCount = S * FF;
        cl_kernel siluBackward = runtime_.kernel("silu_multiply_backward");
        runtime_.argument(siluBackward, 0, retainedGate_[layerIndex]);
        runtime_.argument(siluBackward, 1, retainedUp_[layerIndex]);
        runtime_.argument(siluBackward, 2, dFf_);
        runtime_.argument(siluBackward, 3, tempFf_); // dGate
        runtime_.argument(siluBackward, 4, ff_);     // dUp scratch
        runtime_.argument(siluBackward, 5, ffCount);
        runtime_.enqueue1(siluBackward, ffCount);

        linearDWeight(tempFf_, retainedFfnNorm_[layerIndex],
            layer.gate->gradient, S, D, FF);
        linearDInput(tempFf_, layer.gate->parameter, tempD_, S, D, FF);
        linearDWeight(ff_, retainedFfnNorm_[layerIndex],
            layer.up->gradient, S, D, FF);
        linearDInput(ff_, layer.up->parameter, dContext_, S, D, FF);
        addInPlace(tempD_, dContext_, sd);
        rmsBackward(
            layerAttn_[layerIndex], *layer.ffnNorm,
            tempD_, ffnResidualGradient);
        addInPlace(ffnResidualGradient, current, sd);

        // Attention residual branch.
        linearDWeight(
            ffnResidualGradient, retainedAttentionFlat_[layerIndex],
            layer.o->gradient, S, D, D);
        linearDInput(
            ffnResidualGradient, layer.o->parameter,
            tempD_, S, D, D);
        splitHeads(tempD_, dContext_, HQ);

        // dProbability = dContext * V^T.
        bmmNt(
            dContext_, retainedVRepeat_[layerIndex],
            dProbability_, HQ, S, S, HD, 1.0f);
        // dV = Probability^T * dContext.
        bmmLeftT(
            retainedProbability_[layerIndex], dContext_,
            dVRepeat_, HQ, S, S, HD, 1.0f);

        const int attentionRows = HQ * S, seq = S;
        cl_kernel softmaxBackward =
            runtime_.kernel("causal_softmax_backward");
        runtime_.argument(
            softmaxBackward, 0, retainedProbability_[layerIndex]);
        runtime_.argument(softmaxBackward, 1, dProbability_);
        runtime_.argument(softmaxBackward, 2, attentionRows);
        runtime_.argument(softmaxBackward, 3, seq);
        runtime_.enqueue1(softmaxBackward, attentionRows);

        bmmNn(
            dProbability_, retainedKRepeat_[layerIndex], dQ_,
            HQ, S, S, HD, attentionScale);
        bmmLeftT(
            dProbability_, retainedQRoPe_[layerIndex], dKRepeat_,
            HQ, S, S, HD, attentionScale);
        reduceGqa(dKRepeat_, dK_);
        reduceGqa(dVRepeat_, dV_);
        ropeBackward(dQ_, q_, HQ);
        ropeBackward(dK_, k_, HKV);

        // Projection weight/input gradients need the attention RMSNorm output,
        // which is intentionally recomputed instead of retained per layer.
        rmsForward(layerInput_[layerIndex], *layer.attnNorm, norm_);
        mergeHeads(q_, attentionFlat_, HQ);
        linearDWeight(
            attentionFlat_, norm_, layer.q->gradient, S, D, D);
        // current is no longer needed after the FFN residual was accumulated;
        // reuse it as the deterministic sum of Q/K/V input gradients.
        linearDInput(
            attentionFlat_, layer.q->parameter, current, S, D, D);

        mergeHeads(k_, attentionFlat_, HKV);
        linearDWeight(
            attentionFlat_, norm_, layer.k->gradient, S, D, HKV * HD);
        linearDInput(
            attentionFlat_, layer.k->parameter, tempD_, S, D, HKV * HD);
        addInPlace(current, tempD_, sd);

        mergeHeads(dV_, attentionFlat_, HKV);
        linearDWeight(
            attentionFlat_, norm_, layer.v->gradient, S, D, HKV * HD);
        linearDInput(
            attentionFlat_, layer.v->parameter, tempD_, S, D, HKV * HD);
        addInPlace(current, tempD_, sd);

        rmsBackward(
            layerInput_[layerIndex], *layer.attnNorm,
            current, tempD_);
        add(ffnResidualGradient, tempD_, current, sd);
    }

    // Deterministic token-position reduction, adding the embedding-lookup
    // contribution to the already-written tied LM-head contribution.
    cl_kernel embeddingAdd = runtime_.kernel("embedding_add");
    const int unique = uniqueTokenCount_, d = D;
    runtime_.argument(embeddingAdd, 0, embedding_->gradient);
    runtime_.argument(embeddingAdd, 1, current);
    runtime_.argument(embeddingAdd, 2, uniqueTokenIds_);
    runtime_.argument(embeddingAdd, 3, uniquePositions_);
    runtime_.argument(embeddingAdd, 4, uniqueOffsets_);
    runtime_.argument(embeddingAdd, 5, unique);
    runtime_.argument(embeddingAdd, 6, d);
    runtime_.enqueue1(
        embeddingAdd, static_cast<size_t>(uniqueTokenCount_) * D);
}

void NativeTrainer::computeGlobalNorm() {
    int offset = 0;
    for (const auto& pair : slots_) {
        const auto& value = pair.second;
        req(value.elements <= static_cast<size_t>(std::numeric_limits<int>::max()),
            "gradient tensor too large for OpenCL reduction");
        const int count = static_cast<int>(value.elements);
        cl_kernel partial = runtime_.kernel("sumsq_partial");
        runtime_.argument(partial, 0, value.gradient);
        runtime_.argument(partial, 1, normPartials_);
        runtime_.argument(partial, 2, count);
        runtime_.argument(partial, 3, offset);
        runtime_.enqueue1(partial, REDUCE_ITEMS, REDUCE_LOCAL);
        offset += REDUCE_GROUPS;
    }
    req(offset == static_cast<int>(slots_.size()) * REDUCE_GROUPS,
        "global norm partial count mismatch");
    cl_kernel finish = runtime_.kernel("finish_norm");
    runtime_.argument(finish, 0, normPartials_);
    runtime_.argument(finish, 1, globalNorm_);
    runtime_.argument(finish, 2, offset);
    runtime_.enqueue1(finish, 1, 1);
}

void NativeTrainer::adamStep() {
    ++optimizerStep_;
    req(optimizerStep_ <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
        "optimizer step overflow");
    req(std::isfinite(currentLearningRate_) && currentLearningRate_ > 0.0f,
        "learning rate was not initialized");
    const float lr = currentLearningRate_;
    const float beta1 = static_cast<float>(bundle_.adam.beta1);
    const float beta2 = static_cast<float>(bundle_.adam.beta2);
    const float eps = static_cast<float>(bundle_.adam.eps);
    const float biasCorrection1 = 1.0f -
        std::pow(beta1, static_cast<float>(optimizerStep_));
    const float biasCorrection2 = 1.0f -
        std::pow(beta2, static_cast<float>(optimizerStep_));
    for (auto& pair : slots_) {
        auto& value = pair.second;
        req(value.elements <= static_cast<size_t>(std::numeric_limits<int>::max()),
            "AdamW tensor too large");
        const int count = static_cast<int>(value.elements);
        cl_kernel update = runtime_.kernel("adamw_update");
        runtime_.argument(update, 0, value.parameter);
        runtime_.argument(update, 1, value.gradient);
        runtime_.argument(update, 2, value.moment1);
        runtime_.argument(update, 3, value.moment2);
        runtime_.argument(update, 4, globalNorm_);
        runtime_.argument(update, 5, count);
        runtime_.argument(update, 6, lr);
        runtime_.argument(update, 7, beta1);
        runtime_.argument(update, 8, beta2);
        runtime_.argument(update, 9, eps);
        runtime_.argument(update, 10, value.weightDecay);
        runtime_.argument(update, 11, biasCorrection1);
        runtime_.argument(update, 12, biasCorrection2);
        runtime_.enqueue1(update, value.elements);
    }
}

void NativeTrainer::fullTrainingStep() {
    forward();
    backward();
    computeGlobalNorm();
    adamStep();
}

std::vector<float> NativeTrainer::gather(
    cl_mem source, const std::vector<int>& indices) {
    req(!indices.empty() && indices.size() <= 4096,
        "invalid compact probe count");
    runtime_.write(
        probeIndices_, indices.data(), indices.size() * sizeof(int32_t));
    cl_kernel kernel = runtime_.kernel("gather_values");
    const int count = static_cast<int>(indices.size());
    runtime_.argument(kernel, 0, source);
    runtime_.argument(kernel, 1, probeIndices_);
    runtime_.argument(kernel, 2, probeOutput_);
    runtime_.argument(kernel, 3, count);
    runtime_.enqueue1(kernel, indices.size());
    std::vector<float> result(indices.size());
    runtime_.read(
        probeOutput_, result.data(), result.size() * sizeof(float));
    return result;
}

ProbeError NativeTrainer::validateWeightLoad() {
    ProbeError error;
    for (const auto& pair : slots_) {
        const auto& host = bundle_.tensor(pair.first).data;
        const int last = static_cast<int>(host.size() - 1);
        const std::vector<int> indices = {0, last / 2, last};
        const auto got = gather(pair.second.parameter, indices);
        for (size_t i = 0; i < indices.size(); ++i) {
            const double reference = host[indices[i]];
            const double abs = std::abs(static_cast<double>(got[i]) - reference);
            const double rel = relativeError(got[i], reference);
            if (abs > error.maxAbs || error.index < 0) {
                error.maxAbs = abs;
                error.slot = pair.first;
                error.index = indices[i];
                error.reference = reference;
                error.got = got[i];
            }
            error.maxRel = std::max(error.maxRel, rel);
        }
    }
    return error;
}

ForwardGate NativeTrainer::checkForward() {
    runtime_.finish();
    ForwardGate gate;
    float loss = 0.0f;
    runtime_.read(loss_, &loss, sizeof(loss));
    gate.loss = loss;
    gate.lossAbs = std::abs(gate.loss - bundle_.reference.loss);

    const auto& specs = bundle_.manifest["reference"]["logit_probe"];
    req(specs.IsArray() && !specs.Empty(), "reference logit probes missing");
    std::vector<int> indices;
    std::vector<double> references;
    std::vector<int> positions;
    std::vector<int> tokens;
    for (const auto& spec : specs.GetArray()) {
        const int position = spec["position"].GetInt();
        const int token = spec["token"].GetInt();
        indices.push_back(position * V + token);
        references.push_back(spec["value"].GetDouble());
        positions.push_back(position);
        tokens.push_back(token);
    }
    const auto got = gather(logits_, indices);
    for (size_t i = 0; i < got.size(); ++i) {
        const double abs = std::abs(static_cast<double>(got[i]) - references[i]);
        const double rel = relativeError(got[i], references[i]);
        if (abs > gate.logits.maxAbs || gate.logits.index < 0) {
            gate.logits.maxAbs = abs;
            gate.logits.maxRel = rel;
            gate.logits.slot = "logits";
            gate.logits.index = indices[i];
            gate.logits.reference = references[i];
            gate.logits.got = got[i];
            gate.position = positions[i];
            gate.token = tokens[i];
        }
        gate.logits.maxRel = std::max(gate.logits.maxRel, rel);
    }
    gate.pass = std::isfinite(gate.loss) &&
        std::isfinite(gate.logits.maxAbs) &&
        gate.lossAbs <= 2.0e-3 && gate.logits.maxAbs <= 5.0e-3;
    return gate;
}

BackwardGate NativeTrainer::checkBackward() {
    runtime_.finish();
    BackwardGate gate;
    float norm = 0.0f;
    runtime_.read(globalNorm_, &norm, sizeof(norm));
    gate.norm = norm;
    gate.normRel = relativeError(gate.norm, bundle_.reference.globalGradNorm);

    const auto& all = bundle_.manifest["reference"]["gradient"];
    req(all.IsObject(), "reference gradient map missing");
    for (const auto& pair : slots_) {
        req(all.HasMember(pair.first.c_str()),
            "reference gradient slot missing " + pair.first);
        const auto& spec = all[pair.first.c_str()];
        const auto& manifestIndices = spec["probe_indices"];
        const auto& manifestValues = spec["probe_values"];
        req(manifestIndices.Size() == manifestValues.Size() &&
            manifestIndices.Size() > 0,
            "reference gradient probe size mismatch " + pair.first);
        std::vector<int> indices;
        std::vector<double> references;
        for (rapidjson::SizeType i = 0; i < manifestIndices.Size(); ++i) {
            indices.push_back(manifestIndices[i].GetInt());
            references.push_back(manifestValues[i].GetDouble());
        }
        const auto got = gather(pair.second.gradient, indices);
        for (size_t i = 0; i < got.size(); ++i) {
            const double abs = std::abs(static_cast<double>(got[i]) - references[i]);
            const double rel = relativeError(got[i], references[i]);
            if (abs > gate.gradient.maxAbs || gate.gradient.index < 0) {
                gate.gradient.maxAbs = abs;
                gate.gradient.slot = pair.first;
                gate.gradient.index = indices[i];
                gate.gradient.reference = references[i];
                gate.gradient.got = got[i];
            }
            gate.gradient.maxRel = std::max(gate.gradient.maxRel, rel);
        }
    }
    gate.pass = std::isfinite(gate.norm) &&
        std::isfinite(gate.gradient.maxAbs) &&
        gate.normRel <= 2.0e-2 && gate.gradient.maxAbs <= 5.0e-3;
    return gate;
}

AdamGate NativeTrainer::checkAdam() {
    runtime_.finish();
    AdamGate gate;
    const auto& all = bundle_.manifest["reference"]["adamw_step1"];
    req(all.IsObject(), "reference AdamW map missing");
    for (const auto& pair : slots_) {
        req(all.HasMember(pair.first.c_str()),
            "reference AdamW slot missing " + pair.first);
        const auto& spec = all[pair.first.c_str()];
        const auto& manifestIndices = spec["probe_indices"];
        const auto& manifestValues = spec["after"];
        req(manifestIndices.Size() == manifestValues.Size() &&
            manifestIndices.Size() > 0,
            "reference AdamW probe size mismatch " + pair.first);
        std::vector<int> indices;
        std::vector<double> references;
        for (rapidjson::SizeType i = 0; i < manifestIndices.Size(); ++i) {
            indices.push_back(manifestIndices[i].GetInt());
            references.push_back(manifestValues[i].GetDouble());
        }
        const auto got = gather(pair.second.parameter, indices);
        for (size_t i = 0; i < got.size(); ++i) {
            const double abs = std::abs(static_cast<double>(got[i]) - references[i]);
            const double rel = relativeError(got[i], references[i]);
            if (abs > gate.parameter.maxAbs || gate.parameter.index < 0) {
                gate.parameter.maxAbs = abs;
                gate.parameter.slot = pair.first;
                gate.parameter.index = indices[i];
                gate.parameter.reference = references[i];
                gate.parameter.got = got[i];
            }
            gate.parameter.maxRel = std::max(gate.parameter.maxRel, rel);
        }
    }
    gate.pass = std::isfinite(gate.parameter.maxAbs) &&
        gate.parameter.maxAbs <= 5.0e-4;
    return gate;
}

std::vector<StateProbe> NativeTrainer::captureStateProbes() {
    std::vector<StateProbe> result;
    result.reserve(slots_.size() * 3);
    for (const auto& pair : slots_) {
        const int last = static_cast<int>(pair.second.elements - 1);
        const std::vector<int> indices = {0, last / 2, last};
        const auto parameters = gather(pair.second.parameter, indices);
        const auto moment1 = gather(pair.second.moment1, indices);
        const auto moment2 = gather(pair.second.moment2, indices);
        for (size_t i = 0; i < indices.size(); ++i) {
            result.push_back({
                pair.first, indices[i],
                parameters[i], moment1[i], moment2[i]});
        }
    }
    return result;
}

template <class T>
void fileWrite(FILE* file, const T& value, const std::string& label) {
    req(std::fwrite(&value, sizeof(T), 1, file) == 1,
        "checkpoint write failed: " + label);
}

void fileWriteBytes(
    FILE* file, const void* data, size_t size, const std::string& label) {
    req(size == 0 || std::fwrite(data, 1, size, file) == size,
        "checkpoint write failed: " + label);
}

template <class T>
void fileRead(FILE* file, T* value, const std::string& label) {
    req(std::fread(value, sizeof(T), 1, file) == 1,
        "checkpoint read failed: " + label);
}

void fileReadBytes(
    FILE* file, void* data, size_t size, const std::string& label) {
    req(size == 0 || std::fread(data, 1, size, file) == size,
        "checkpoint read failed: " + label);
}

void writeString(FILE* file, const std::string& value, const std::string& label) {
    req(value.size() <= std::numeric_limits<uint32_t>::max(),
        "checkpoint string too large: " + label);
    const uint32_t length = static_cast<uint32_t>(value.size());
    fileWrite(file, length, label + ".length");
    fileWriteBytes(file, value.data(), value.size(), label);
}

std::string readString(FILE* file, const std::string& label) {
    uint32_t length = 0;
    fileRead(file, &length, label + ".length");
    req(length <= 4096, "checkpoint string length invalid: " + label);
    std::string value(length, '\0');
    if (length) fileReadBytes(file, &value[0], value.size(), label);
    return value;
}

void NativeTrainer::saveCheckpoint(
    const std::string& path, size_t* bytesWritten) {
    const std::string temporary = path + ".tmp";
    FILE* file = std::fopen(temporary.c_str(), "wb");
    req(file != nullptr,
        "cannot create checkpoint " + temporary + ": " + std::strerror(errno));
    try {
        const char magic[8] = {'A', 'T', 'N', 'C', 'L', '0', '1', '\0'};
        fileWriteBytes(file, magic, sizeof(magic), "magic");
        const uint32_t version = 1;
        const uint32_t endian = 0x01020304u;
        const uint32_t tensorCount = static_cast<uint32_t>(slots_.size());
        const uint32_t reserved = 0;
        const uint64_t parameterCount =
            static_cast<uint64_t>(bundle_.parameterCount);
        fileWrite(file, version, "version");
        fileWrite(file, endian, "endian");
        fileWrite(file, tensorCount, "tensor_count");
        fileWrite(file, reserved, "reserved");
        fileWrite(file, optimizerStep_, "optimizer_step");
        fileWrite(file, parameterCount, "parameter_count");
        const std::array<uint32_t, 8> geometry = {
            S, V, D, LAYERS, HQ, HKV, HD, FF};
        for (uint32_t value : geometry) fileWrite(file, value, "geometry");
        fileWrite(file, bundle_.adam.beta1, "beta1");
        fileWrite(file, bundle_.adam.beta2, "beta2");
        fileWrite(file, bundle_.adam.eps, "eps");
        fileWrite(file, bundle_.adam.gateLr, "learning_rate");
        writeString(file, ANDROID_TRAINER_GIT_COMMIT, "commit");
        writeString(file, bundle_.checkpointSha256, "source_checkpoint_sha256");
        writeString(file, bundle_.modelStateSha256, "source_model_sha256");

        constexpr size_t chunkBytes = 1u << 20;
        std::vector<unsigned char> chunk(chunkBytes);
        auto writeDeviceBuffer = [&](cl_mem buffer, size_t bytes,
                                     const std::string& label) {
            size_t offset = 0;
            while (offset < bytes) {
                const size_t count = std::min(chunk.size(), bytes - offset);
                runtime_.read(buffer, chunk.data(), count, offset);
                fileWriteBytes(file, chunk.data(), count, label);
                offset += count;
            }
        };

        for (const auto& pair : slots_) {
            const auto& value = pair.second;
            writeString(file, value.name, "slot_name");
            const uint32_t rank = static_cast<uint32_t>(value.shape.size());
            fileWrite(file, rank, "slot_rank");
            for (int dimension : value.shape) {
                const uint32_t dim = static_cast<uint32_t>(dimension);
                fileWrite(file, dim, "slot_dimension");
            }
            const uint64_t elements = static_cast<uint64_t>(value.elements);
            const double weightDecay = value.weightDecay;
            fileWrite(file, elements, "slot_elements");
            fileWrite(file, weightDecay, "slot_weight_decay");
            const size_t bytes = bytesFor(value.elements);
            writeDeviceBuffer(value.parameter, bytes, value.name + ".parameter");
            writeDeviceBuffer(value.moment1, bytes, value.name + ".moment1");
            writeDeviceBuffer(value.moment2, bytes, value.name + ".moment2");
        }
        req(std::fflush(file) == 0, "checkpoint fflush failed");
        req(::fsync(::fileno(file)) == 0, "checkpoint fsync failed");
        const long end = std::ftell(file);
        req(end >= 0, "checkpoint ftell failed");
        req(std::fclose(file) == 0, "checkpoint fclose failed");
        file = nullptr;
        req(::rename(temporary.c_str(), path.c_str()) == 0,
            "checkpoint atomic rename failed: " + std::string(std::strerror(errno)));
        if (bytesWritten) *bytesWritten = static_cast<size_t>(end);
    } catch (...) {
        if (file) std::fclose(file);
        std::remove(temporary.c_str());
        throw;
    }
}

void NativeTrainer::loadCheckpoint(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "rb");
    req(file != nullptr,
        "cannot open native checkpoint " + path + ": " + std::strerror(errno));
    try {
        char magic[8] = {};
        fileReadBytes(file, magic, sizeof(magic), "magic");
        const char expected[8] = {'A', 'T', 'N', 'C', 'L', '0', '1', '\0'};
        req(std::memcmp(magic, expected, sizeof(magic)) == 0,
            "native checkpoint magic mismatch");
        uint32_t version = 0, endian = 0, tensorCount = 0, reserved = 0;
        uint64_t step = 0, parameterCount = 0;
        fileRead(file, &version, "version");
        fileRead(file, &endian, "endian");
        fileRead(file, &tensorCount, "tensor_count");
        fileRead(file, &reserved, "reserved");
        fileRead(file, &step, "optimizer_step");
        fileRead(file, &parameterCount, "parameter_count");
        req(version == 1 && endian == 0x01020304u && reserved == 0,
            "native checkpoint header mismatch");
        req(tensorCount == slots_.size() &&
            parameterCount == static_cast<uint64_t>(bundle_.parameterCount),
            "native checkpoint tensor/parameter count mismatch");
        const std::array<uint32_t, 8> expectedGeometry = {
            S, V, D, LAYERS, HQ, HKV, HD, FF};
        for (uint32_t expectedValue : expectedGeometry) {
            uint32_t value = 0;
            fileRead(file, &value, "geometry");
            req(value == expectedValue, "native checkpoint geometry mismatch");
        }
        double beta1 = 0.0, beta2 = 0.0, eps = 0.0, lr = 0.0;
        fileRead(file, &beta1, "beta1");
        fileRead(file, &beta2, "beta2");
        fileRead(file, &eps, "eps");
        fileRead(file, &lr, "learning_rate");
        req(beta1 == bundle_.adam.beta1 && beta2 == bundle_.adam.beta2 &&
            eps == bundle_.adam.eps && lr == bundle_.adam.gateLr,
            "native checkpoint optimizer contract mismatch");
        const std::string commit = readString(file, "commit");
        const std::string checkpointSha =
            readString(file, "source_checkpoint_sha256");
        const std::string modelSha = readString(file, "source_model_sha256");
        req(commit == ANDROID_TRAINER_GIT_COMMIT,
            "native checkpoint build commit mismatch");
        req(checkpointSha == bundle_.checkpointSha256 &&
            modelSha == bundle_.modelStateSha256,
            "native checkpoint source identity mismatch");

        constexpr size_t chunkBytes = 1u << 20;
        std::vector<unsigned char> chunk(chunkBytes);
        auto readDeviceBuffer = [&](cl_mem buffer, size_t bytes,
                                    const std::string& label) {
            size_t offset = 0;
            while (offset < bytes) {
                const size_t count = std::min(chunk.size(), bytes - offset);
                fileReadBytes(file, chunk.data(), count, label);
                runtime_.write(buffer, chunk.data(), count, offset);
                offset += count;
            }
        };

        for (auto& pair : slots_) {
            auto& value = pair.second;
            const std::string name = readString(file, "slot_name");
            req(name == value.name,
                "native checkpoint slot order/name mismatch: expected " +
                value.name + " got " + name);
            uint32_t rank = 0;
            fileRead(file, &rank, "slot_rank");
            req(rank == value.shape.size(), "native checkpoint slot rank mismatch");
            for (int expectedDimension : value.shape) {
                uint32_t dimension = 0;
                fileRead(file, &dimension, "slot_dimension");
                req(dimension == static_cast<uint32_t>(expectedDimension),
                    "native checkpoint slot shape mismatch: " + value.name);
            }
            uint64_t elements = 0;
            double weightDecay = 0.0;
            fileRead(file, &elements, "slot_elements");
            fileRead(file, &weightDecay, "slot_weight_decay");
            req(elements == value.elements &&
                weightDecay == static_cast<double>(value.weightDecay),
                "native checkpoint slot metadata mismatch: " + value.name);
            const size_t bytes = bytesFor(value.elements);
            readDeviceBuffer(value.parameter, bytes, value.name + ".parameter");
            readDeviceBuffer(value.moment1, bytes, value.name + ".moment1");
            readDeviceBuffer(value.moment2, bytes, value.name + ".moment2");
        }
        req(std::fgetc(file) == EOF, "native checkpoint has trailing bytes");
        req(std::fclose(file) == 0, "native checkpoint fclose failed");
        file = nullptr;
        optimizerStep_ = step;
        runtime_.finish();
    } catch (...) {
        if (file) std::fclose(file);
        throw;
    }
}

CheckpointGate NativeTrainer::checkpointRoundTrip() {
    CheckpointGate gate;
    gate.path = workDirectory_ + "/model0001-native-opencl.atnckpt";
    const auto before = captureStateProbes();
    saveCheckpoint(gate.path, &gate.bytes);

    // Clear all persisted state before reload. This makes the proof a genuine
    // deserialize/upload round trip rather than a comparison with untouched
    // live buffers.
    for (auto& pair : slots_) {
        auto& value = pair.second;
        const size_t bytes = bytesFor(value.elements);
        runtime_.zero(value.parameter, bytes);
        runtime_.zero(value.moment1, bytes);
        runtime_.zero(value.moment2, bytes);
    }
    runtime_.finish();
    loadCheckpoint(gate.path);
    const auto after = captureStateProbes();
    req(after.size() == before.size(), "checkpoint probe count changed");
    for (size_t i = 0; i < before.size(); ++i) {
        req(before[i].slot == after[i].slot &&
            before[i].index == after[i].index,
            "checkpoint probe identity changed");
        gate.maxAbs = std::max({
            gate.maxAbs,
            std::abs(static_cast<double>(before[i].parameter) - after[i].parameter),
            std::abs(static_cast<double>(before[i].moment1) - after[i].moment1),
            std::abs(static_cast<double>(before[i].moment2) - after[i].moment2),
        });
        gate.probes += 3;
    }
    gate.pass = gate.maxAbs == 0.0 && gate.probes > 0 && gate.bytes > 0;
    return gate;
}

BenchmarkGate NativeTrainer::benchmark(
    const std::function<double()>& cpuBaseline) {
    BenchmarkGate gate;

    // Gate 4 performs one real AdamW update and gate 5 reloads that mutated
    // checkpoint.  The CPU reference benchmark starts from immutable CPT-v2
    // weights with fresh zero moments, so reset before warmup/timing to make
    // the sustained comparison truly apples-to-apples.
    resetToSourceState();
    req(optimizerStep_ == 0, "benchmark reset did not restore optimizer step");

    // Gates 1-5 already prove correctness. A missing/invalid CPU speed
    // baseline must not block the native GPU benchmark itself.
    if (cpuBaseline) {
        try {
            gate.cpuTokensPerSecond = cpuBaseline();
        } catch (...) {
            gate.cpuTokensPerSecond = 0.0;
        }
    }

    fullTrainingStep();
    runtime_.finish();
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < BENCH_STEPS; ++i) fullTrainingStep();
    // This finish is deliberately before the stop timestamp. Queue submission
    // latency is never reported as completed training throughput.
    runtime_.finish();
    const auto stopped = std::chrono::steady_clock::now();
    gate.seconds = std::chrono::duration<double>(stopped - started).count();
    gate.tokensPerSecond = BENCH_STEPS * S /
        std::max(gate.seconds, 1.0e-9);
    gate.ratio = gate.cpuTokensPerSecond > 0.0
        ? gate.tokensPerSecond / gate.cpuTokensPerSecond
        : 0.0;
    float loss = 0.0f;
    runtime_.read(loss_, &loss, sizeof(loss));
    gate.finalLoss = loss;
    gate.useful = std::isfinite(gate.ratio) && gate.ratio >= 1.5;
    gate.canonical = std::isfinite(gate.ratio) && gate.ratio >= 2.0;
    gate.pass = std::isfinite(gate.finalLoss) &&
        std::isfinite(gate.tokensPerSecond) && gate.tokensPerSecond > 0.0;
    return gate;
}

StageProfile NativeTrainer::profileStages() {
    StageProfile profile;

    // Diagnostic-only path. The official sustained benchmark has already
    // completed before this method is called. We deliberately reset again so
    // the profiler cannot inherit benchmark state or affect acceptance data.
    resetToSourceState();

    // One untimed warmup, then reset again so profiled steps start at the same
    // optimizer state while all kernels/runtime objects are already hot.
    fullTrainingStep();
    runtime_.finish();
    resetToSourceState();

    auto measure = [&](auto&& fn) {
        const auto started = std::chrono::steady_clock::now();
        fn();
        runtime_.finish();
        const auto stopped = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(stopped - started).count();
    };

    for (int i = 0; i < profile.profiledSteps; ++i) {
        profile.forwardSeconds += measure([&]() { forward(); });
        profile.backwardSeconds += measure([&]() { backward(); });
        profile.gradNormSeconds += measure([&]() { computeGlobalNorm(); });
        profile.adamwSeconds += measure([&]() { adamStep(); });
    }

    profile.totalSeconds =
        profile.forwardSeconds +
        profile.backwardSeconds +
        profile.gradNormSeconds +
        profile.adamwSeconds;

    profile.pass =
        std::isfinite(profile.forwardSeconds) && profile.forwardSeconds > 0.0 &&
        std::isfinite(profile.backwardSeconds) && profile.backwardSeconds > 0.0 &&
        std::isfinite(profile.gradNormSeconds) && profile.gradNormSeconds > 0.0 &&
        std::isfinite(profile.adamwSeconds) && profile.adamwSeconds > 0.0 &&
        std::isfinite(profile.totalSeconds) && profile.totalSeconds > 0.0;
    return profile;
}

std::string NativeTrainer::profileKernelsJson() {
    // This is a second, diagnostic-only profiling pass. It uses a dedicated
    // CL_QUEUE_PROFILING_ENABLE queue so official sustained timing above stays
    // on the original non-profiling queue.
    resetToSourceState();
    runtime_.beginDiagnosticQueue();
    try {
        // Warm the dedicated queue without collecting events, then restore the
        // immutable source state so capture starts from optimizer step zero.
        fullTrainingStep();
        runtime_.finish();
        resetToSourceState();

        // Build the exact forward activations and dLogits first without
        // collecting events. Capture only backward() so the kernel ranking
        // answers the already-proven 66.8% backward bottleneck specifically.
        forward();
        runtime_.finish();

        runtime_.beginKernelCapture();
        backward();
        const auto totals = runtime_.endKernelCapture();
        runtime_.endDiagnosticQueue();

        struct Row {
            std::string name;
            int count = 0;
            double seconds = 0.0;
        };
        std::vector<Row> rows;
        double totalKernelSeconds = 0.0;
        int totalEvents = 0;
        for (const auto& pair : totals) {
            rows.push_back({pair.first, pair.second.first, pair.second.second});
            totalKernelSeconds += pair.second.second;
            totalEvents += pair.second.first;
        }
        std::sort(rows.begin(), rows.end(),
            [](const Row& a, const Row& b) { return a.seconds > b.seconds; });

        std::ostringstream out;
        out << "{\"pass\":true"
            << ",\"diagnostic_only\":true"
            << ",\"used_for_acceptance\":false"
            << ",\"queue_profiling_enabled\":true"
            << ",\"profiled_backward_passes\":1"
            << ",\"forward_prerun_profiled\":false"
            << ",\"event_count\":" << totalEvents
            << ",\"total_kernel_seconds\":" << totalKernelSeconds
            << ",\"kernels\":[";
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i) out << ",";
            const double fraction = totalKernelSeconds > 0.0
                ? rows[i].seconds / totalKernelSeconds : 0.0;
            out << "{\"name\":\"" << jsonEscape(rows[i].name)
                << "\",\"count\":" << rows[i].count
                << ",\"seconds\":" << rows[i].seconds
                << ",\"fraction\":" << fraction << "}";
        }
        out << "]}";
        return out.str();
    } catch (...) {
        // Restore the official queue even if diagnostic profiling fails.
        // Diagnostic failure must never invalidate the already-completed gate.
        try {
            if (runtime_.kernelCaptureActive) {
                (void)runtime_.endKernelCapture();
            }
        } catch (...) {}
        try { runtime_.endDiagnosticQueue(); } catch (...) {}
        throw;
    }
}

std::string NativeTrainer::memoryJson() const {
    const size_t total = persistentBytes_ + activationBytes_ + workspaceBytes_;
    std::ostringstream out;
    out << "{\"persistent_parameter_gradient_adam_bytes\":"
        << persistentBytes_
        << ",\"retained_activation_bytes\":" << activationBytes_
        << ",\"reused_workspace_bytes\":" << workspaceBytes_
        << ",\"estimated_total_opencl_bytes\":" << total
        << ",\"estimated_total_mib\":"
        << static_cast<double>(total) / (1024.0 * 1024.0) << "}";
    return out.str();
}

std::string probeErrorJson(const ProbeError& error) {
    std::ostringstream out;
    out << "{\"slot\":\"" << jsonEscape(error.slot)
        << "\",\"index\":" << error.index
        << ",\"reference\":" << std::setprecision(17) << error.reference
        << ",\"got\":" << error.got
        << ",\"max_abs_error\":" << error.maxAbs
        << ",\"max_rel_error\":" << error.maxRel << "}";
    return out.str();
}

NativeGateResult NativeTrainer::run(
    const std::function<double()>& cpuBaseline) {
    initializeSlots();
    initializeInputs();
    initializeActivations();
    const ProbeError weightError = validateWeightLoad();
    const bool weightPass = std::isfinite(weightError.maxAbs) &&
        weightError.maxAbs == 0.0;
    mark(weightPass
        ? "native:gate1:weight_load:pass"
        : "native:gate1:weight_load:fail");

    std::string forwardJson = "null";
    std::string backwardJson = "null";
    std::string adamJson = "null";
    std::string checkpointJson = "null";
    std::string benchmarkJson = "null";
    std::string profileJson = "null";
    std::string firstFailure;
    std::string firstOperator;
    std::string firstProbe;

    auto weightJson = [&]() {
        std::ostringstream out;
        out << "{\"pass\":" << (weightPass ? "true" : "false")
            << ",\"tensor_count\":" << slots_.size()
            << ",\"parameter_count\":" << bundle_.parameterCount
            << ",\"probe_count\":" << slots_.size() * 3
            << ",\"max_probe_abs_error\":" << weightError.maxAbs
            << ",\"worst\":" << probeErrorJson(weightError) << "}";
        return out.str();
    };

    auto report = [&](bool pass, const std::string& status) {
        std::ostringstream out;
        out << "{\"status\":\"" << status
            << "\",\"schema\":\"model0001_native_opencl_gate_report_v1\""
            << ",\"backend\":\"PURE_OPENCL_C_1_2_FP32_BUFFER\""
            << ",\"commit\":\"" << jsonEscape(ANDROID_TRAINER_GIT_COMMIT)
            << "\",\"device\":" << runtime_.json()
            << ",\"runtime_lifetime\":\"PROCESS_LONG_NO_TEARDOWN\""
            << ",\"kernel_build_options\":[\"-cl-std=CL1.2\"]"
            << ",\"constraints\":{\"mnn_gpu\":false,\"vulkan\":false"
            << ",\"image_tensors\":false,\"fp16\":false"
            << ",\"fast_math\":false,\"float_atomics\":false"
            << ",\"generic_autograd\":false,\"cpu_fallback\":false}"
            << ",\"weight_load\":" << weightJson()
            << ",\"forward_parity\":" << forwardJson
            << ",\"backward_parity\":" << backwardJson
            << ",\"adamw_parity\":" << adamJson
            << ",\"checkpoint_verification\":" << checkpointJson
            << ",\"memory_estimates\":" << memoryJson()
            << ",\"sustained_benchmark\":" << benchmarkJson
            << ",\"performance_profile\":" << profileJson
            << ",\"first_failing_stage\":"
            << (firstFailure.empty()
                ? "null"
                : "\"" + jsonEscape(firstFailure) + "\"")
            << ",\"first_failing_operator\":"
            << (firstOperator.empty()
                ? "null"
                : "\"" + jsonEscape(firstOperator) + "\"")
            << ",\"first_failing_probe\":"
            << (firstProbe.empty()
                ? "null"
                : "\"" + jsonEscape(firstProbe) + "\"")
            << ",\"pass\":" << (pass ? "true" : "false") << "}";
        return NativeGateResult{pass, out.str()};
    };

    if (!weightPass) {
        firstFailure = "GPU_WEIGHT_LOAD";
        firstOperator = "clEnqueueWriteBuffer/readback";
        firstProbe = weightError.slot + "[" +
            std::to_string(weightError.index) + "]";
        return report(false, "FAIL_GPU_WEIGHT_LOAD");
    }

    mark("native:gate2:forward:start");
    forward();
    const ForwardGate forwardGate = checkForward();
    {
        std::ostringstream out;
        out << "{\"pass\":" << (forwardGate.pass ? "true" : "false")
            << ",\"loss\":" << std::setprecision(17) << forwardGate.loss
            << ",\"reference_loss\":" << bundle_.reference.loss
            << ",\"loss_abs_error\":" << forwardGate.lossAbs
            << ",\"loss_abs_threshold\":0.002"
            << ",\"max_logit_probe_abs_error\":"
            << forwardGate.logits.maxAbs
            << ",\"logit_abs_threshold\":0.005"
            << ",\"worst_position\":" << forwardGate.position
            << ",\"worst_token\":" << forwardGate.token
            << ",\"worst\":" << probeErrorJson(forwardGate.logits) << "}";
        forwardJson = out.str();
    }
    mark(forwardGate.pass
        ? "native:gate2:forward:pass"
        : "native:gate2:forward:fail");
    if (!forwardGate.pass) {
        firstFailure = "FULL_NATIVE_FORWARD";
        firstOperator = "locked_logit_probe_or_cross_entropy";
        firstProbe = "position=" + std::to_string(forwardGate.position) +
            ",token=" + std::to_string(forwardGate.token);
        return report(false, "FAIL_FULL_NATIVE_FORWARD");
    }

    mark("native:gate3:backward:start");
    backward();
    computeGlobalNorm();
    const BackwardGate backwardGate = checkBackward();
    {
        std::ostringstream out;
        out << "{\"pass\":" << (backwardGate.pass ? "true" : "false")
            << ",\"global_grad_norm\":" << std::setprecision(17)
            << backwardGate.norm
            << ",\"reference_global_grad_norm\":"
            << bundle_.reference.globalGradNorm
            << ",\"grad_norm_rel_error\":" << backwardGate.normRel
            << ",\"grad_norm_rel_threshold\":0.02"
            << ",\"max_grad_probe_abs_error\":"
            << backwardGate.gradient.maxAbs
            << ",\"grad_probe_abs_threshold\":0.005"
            << ",\"worst\":" << probeErrorJson(backwardGate.gradient)
            << ",\"tied_embedding_reduction\":\"LM_HEAD_PLUS_TOKEN_POSITION\"} ";
        backwardJson = out.str();
    }
    mark(backwardGate.pass
        ? "native:gate3:backward:pass"
        : "native:gate3:backward:fail");
    if (!backwardGate.pass) {
        firstFailure = "FULL_NATIVE_BACKWARD";
        firstOperator = "parameter_gradient";
        firstProbe = backwardGate.gradient.slot + "[" +
            std::to_string(backwardGate.gradient.index) + "]";
        return report(false, "FAIL_FULL_NATIVE_BACKWARD");
    }

    mark("native:gate4:fresh_adamw:start");
    adamStep();
    const AdamGate adamGate = checkAdam();
    {
        std::ostringstream out;
        out << "{\"pass\":" << (adamGate.pass ? "true" : "false")
            << ",\"optimizer_step\":" << optimizerStep_
            << ",\"fresh_zero_moments\":true"
            << ",\"beta1\":0.9,\"beta2\":0.95,\"eps\":1e-8"
            << ",\"learning_rate\":0.0001"
            << ",\"max_parameter_probe_abs_error\":"
            << adamGate.parameter.maxAbs
            << ",\"parameter_probe_abs_threshold\":0.0005"
            << ",\"worst\":" << probeErrorJson(adamGate.parameter) << "}";
        adamJson = out.str();
    }
    mark(adamGate.pass
        ? "native:gate4:fresh_adamw:pass"
        : "native:gate4:fresh_adamw:fail");
    if (!adamGate.pass) {
        firstFailure = "FRESH_ADAMW_STEP";
        firstOperator = "clip_plus_decoupled_adamw";
        firstProbe = adamGate.parameter.slot + "[" +
            std::to_string(adamGate.parameter.index) + "]";
        return report(false, "FAIL_FRESH_ADAMW");
    }

    mark("native:gate5:checkpoint:start");
    const CheckpointGate checkpointGate = checkpointRoundTrip();
    {
        std::ostringstream out;
        out << "{\"pass\":" << (checkpointGate.pass ? "true" : "false")
            << ",\"format\":\"android_trainer_native_checkpoint_v1\""
            << ",\"path\":\"" << jsonEscape(checkpointGate.path)
            << "\",\"bytes\":" << checkpointGate.bytes
            << ",\"parameter_m_v_probe_count\":" << checkpointGate.probes
            << ",\"reload_max_abs_error\":" << checkpointGate.maxAbs
            << ",\"cleared_before_reload\":true} ";
        checkpointJson = out.str();
    }
    mark(checkpointGate.pass
        ? "native:gate5:checkpoint:pass"
        : "native:gate5:checkpoint:fail");
    if (!checkpointGate.pass) {
        firstFailure = "CHECKPOINT_PROOF";
        firstOperator = "serialize_clear_reload";
        firstProbe = "parameter_m_v";
        return report(false, "FAIL_CHECKPOINT_PROOF");
    }

    mark("native:gate6:sustained_benchmark:start");
    const BenchmarkGate benchmarkGate = benchmark(cpuBaseline);
    {
        std::ostringstream out;
        out << "{\"pass\":" << (benchmarkGate.pass ? "true" : "false")
            << ",\"warmup_steps\":" << benchmarkGate.warmupSteps
            << ",\"timed_steps\":" << benchmarkGate.timedSteps
            << ",\"target_tokens_per_step\":" << S
            << ",\"reset_to_source_before_benchmark\":true"
            << ",\"starting_optimizer_step\":0"
            << ",\"ending_optimizer_step\":" << (benchmarkGate.warmupSteps + benchmarkGate.timedSteps)
            << ",\"seconds\":" << benchmarkGate.seconds
            << ",\"synchronized_before_stop\":true"
            << ",\"native_tokens_per_second\":"
            << benchmarkGate.tokensPerSecond
            << ",\"cpu_tokens_per_second\":"
            << benchmarkGate.cpuTokensPerSecond
            << ",\"native_vs_cpu_ratio\":" << benchmarkGate.ratio
            << ",\"useful_threshold_1_5x\":"
            << (benchmarkGate.useful ? "true" : "false")
            << ",\"canonical_threshold_2_0x\":"
            << (benchmarkGate.canonical ? "true" : "false")
            << ",\"final_loss\":" << benchmarkGate.finalLoss << "}";
        benchmarkJson = out.str();
    }
    mark(benchmarkGate.pass
        ? "native:gate6:sustained_benchmark:pass"
        : "native:gate6:sustained_benchmark:fail");
    if (!benchmarkGate.pass) {
        firstFailure = "SUSTAINED_BENCHMARK";
        firstOperator = "full_training_step";
        firstProbe = "finite_loss_and_synchronized_tokens_per_second";
        return report(false, "FAIL_SUSTAINED_BENCHMARK");
    }

    // Diagnostic profiling is strictly post-acceptance-measurement. It never
    // feeds the sustained tok/s or pass/fail decision above.
    mark("native:diagnostic_profile:start");
    try {
        const StageProfile profile = profileStages();
        const std::string kernelProfile = profileKernelsJson();
        const double denom = std::max(profile.totalSeconds, 1.0e-12);
        std::ostringstream out;
        out << "{\"pass\":" << (profile.pass ? "true" : "false")
            << ",\"diagnostic_only\":true"
            << ",\"used_for_acceptance\":false"
            << ",\"synchronization_between_stages\":true"
            << ",\"warmup_steps\":" << profile.warmupSteps
            << ",\"profiled_steps\":" << profile.profiledSteps
            << ",\"forward_seconds\":" << profile.forwardSeconds
            << ",\"backward_seconds\":" << profile.backwardSeconds
            << ",\"grad_norm_seconds\":" << profile.gradNormSeconds
            << ",\"adamw_seconds\":" << profile.adamwSeconds
            << ",\"total_seconds\":" << profile.totalSeconds
            << ",\"forward_fraction\":" << (profile.forwardSeconds / denom)
            << ",\"backward_fraction\":" << (profile.backwardSeconds / denom)
            << ",\"grad_norm_fraction\":" << (profile.gradNormSeconds / denom)
            << ",\"adamw_fraction\":" << (profile.adamwSeconds / denom)
            << ",\"kernel_profile\":" << kernelProfile
            << "}";
        profileJson = out.str();
    } catch (const std::exception& error) {
        profileJson = std::string("{\"pass\":false,\"diagnostic_only\":true,") +
            "\"used_for_acceptance\":false,\"error\":\"" +
            jsonEscape(error.what()) + "\"}";
    }
    mark("native:diagnostic_profile:done");
    mark("native:all_gates:pass");
    return report(true, "PASS");
}

}  // namespace

std::string probeNativeOpenClJson() {
    try {
        auto& runtime = processRuntime();
        return std::string("{\"status\":\"PASS\",\"backend\":") +
            "\"PURE_OPENCL_C_1_2_FP32_BUFFER\",\"device\":" +
            runtime.json() +
            ",\"runtime_lifetime\":\"PROCESS_LONG_NO_TEARDOWN\"}";
    } catch (const std::exception& error) {
        return std::string("{\"status\":\"FAIL\",\"backend\":") +
            "\"PURE_OPENCL_C_1_2_FP32_BUFFER\",\"error\":\"" +
            jsonEscape(error.what()) + "\"}";
    }
}

NativeGateResult runNativeModel0001Gate(
    const Bundle& bundle,
    const std::string& workDirectory,
    const std::function<double()>& cpuBaselineTokensPerSecond) {
    static bool completed = false;
    static NativeGateResult cached;
    if (completed) return cached;
    NativeTrainer* trainer = nullptr;
    try {
        // The object and every OpenCL allocation are intentionally retained to
        // process death. This avoids entering the proven-bad Mali teardown path.
        trainer = new NativeTrainer(bundle, workDirectory);
        cached = trainer->run(cpuBaselineTokensPerSecond);
    } catch (const std::exception& error) {
        std::ostringstream out;
        out << "{\"status\":\"FAIL_NATIVE_EXCEPTION\""
            << ",\"schema\":\"model0001_native_opencl_gate_report_v1\""
            << ",\"backend\":\"PURE_OPENCL_C_1_2_FP32_BUFFER\""
            << ",\"commit\":\"" << jsonEscape(ANDROID_TRAINER_GIT_COMMIT)
            << "\",\"first_failing_stage\":\""
            << jsonEscape(trainer ? trainer->currentStage() : "native:initialize")
            << "\""
            << ",\"error\":\"" << jsonEscape(error.what())
            << "\",\"pass\":false}";
        cached = {false, out.str()};
    }
    completed = true;
    return cached;
}

}  // namespace at
