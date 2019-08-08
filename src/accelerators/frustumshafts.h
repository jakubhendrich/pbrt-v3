
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    This file implements the frustum shafts method from the following paper:

    Jakub Hendrich, Adam Pospíšil, Daniel Meister, Jiří Bittner.
    Ray Classification for Accelerated BVH Traversal
    (https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13769)
    Computer Graphics Forum, 38: 49-56, 2019. doi:10.1111/cgf.13769.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_ACCELERATORS_FRUSTUM_SHAFTS_H
#define PBRT_ACCELERATORS_FRUSTUM_SHAFTS_H

#include "pbrt.h"
#include "primitive.h"
#include "bvh.h"
#include "paramset.h"
#include "geometry.h"

#include <unordered_map>
#include <mutex>
#include <atomic>
#include <random>

namespace pbrt {

#ifndef FRUSTUM_SHAFTS_BUILD_STATS
#  define FRUSTUM_SHAFTS_BUILD_STATS 1
#endif

#ifndef FRUSTUM_SHAFTS_TRAVERSAL_STATS
#  define FRUSTUM_SHAFTS_TRAVERSAL_STATS 1
#endif

#ifndef USE_ONLY_OCCUPIED_CELLS_FOR_INSTANCES
#  define USE_ONLY_OCCUPIED_CELLS_FOR_INSTANCES 1
#endif

#ifndef FIND_SHAFT_FOR_OUTLYING_RAYS
#  define FIND_SHAFT_FOR_OUTLYING_RAYS 0
#endif

// Atomic equipped with copy constructor & assignment operators. The copy operations are *not* atomic, though.
// The purpose is to enable the existence of vectors of atomics while using the copy operations with caution
// (in our case, only a single non-concurrent resize() makes use of them).
template <typename T>
struct atomic_copyable
{
    std::atomic<T> _a;

    atomic_copyable(): _a() {}
    atomic_copyable(const std::atomic<T>& a): _a(a.load()) {}
    atomic_copyable(const atomic_copyable& other): _a(other._a.load()) {}
    void operator=(const std::atomic<T>& a) { _a.store(a.load()); }
    void operator=(const atomic_copyable& other) { _a.store(other._a.load()); }
};

class FrustumShaftBuilder;

class FrustumShaftsAccel : public BVHAccel {
  public:
    FrustumShaftsAccel(std::vector<std::shared_ptr<Primitive>> p,
             const ParamSet &ps,
             int maxPrimsInNode = 1,
             SplitMethod splitMethod = SplitMethod::SAH,
             int numCells = 100000,
             int directionResolution = 2,
             bool shaftTraversal = true);
    void setBVHname(const std::string& name) { BVHname = name; }
    bool UseShafts() const { return shaftTraversal; }
    void InitLUT();
    Vector3i ComputeSpatialResolution(const Bounds3f &box, const int cells);
    const FrustumShaftBuilder* GetBuilder() const { return builder; }
    void BuildShafts(const bool isMainBVH = true);
    void InitCalibration();
    void InitBuild();
    void InitRendering();
    bool Intersect(const Ray &ray, SurfaceInteraction *isect) const;
    bool IntersectP(const Ray &ray) const;

  private:
    void InitMultipliers();
    int GetMaxIndex() const;
    int RayToIndex(const Ray &ray) const;
    inline bool PopulateTraversalStack(const Ray &ray, int* nodesToVisit, int& toVisitOffset, int& currentNodeIndex) const;

    enum Phase {
        Calibration,
        Construction,
        Rendering
    };
    int phase;

    std::string BVHname;
    FrustumShaftBuilder* builder;

    // ray origin bounding box (can be scene bounding box; in our case a bit inflated for robustness)
    Bounds3f Bbox;
    int numCells;
    Vector3i originResolution;
    int directionResolution;

    // ray counter / histogram
    mutable std::vector<atomic_copyable<int>> raysInShafts;
    // lookup table for translation from shaft index to offset in candidate lists
    std::vector<int> shaftIndexToCLAddr;
    // concatenated candidate lists, each list consists of: int N (length) + int[N] (node indices)
    std::vector<int> candidateLists;

    Vector3f spatialMult;
    int indexMult[5];

    // use the shafts vs only from-root traversal
    bool shaftTraversal;

    friend class FrustumShaftBuilder;
};

std::shared_ptr<FrustumShaftsAccel> CreateFrustumShaftsAccelerator(
    std::vector<std::shared_ptr<Primitive>> prims, const ParamSet &ps);

namespace frustumshafts {

struct Histogram
{
    float minValue;
    float maxValue;
    std::vector<int> bins;
    int validEntries;
    int totalEntries;

    Histogram():
        minValue(std::numeric_limits<float>::quiet_NaN()),
        maxValue(std::numeric_limits<float>::quiet_NaN()),
        validEntries(0),
        totalEntries(0) {}

    void Init(const int numBins, const float _minValue, const float _maxValue) {
        bins.resize(numBins);
        for (unsigned int i=0; i < bins.size(); i++)
            bins[i] = 0;
        minValue = _minValue;
        maxValue = _maxValue;
        validEntries = 0;
        totalEntries = 0;
    }

    float GetNormalizedValue(const float value) {
        int bin = (int)(bins.size()*(value - minValue)/(maxValue-minValue));
        if (bin < 0)
            bin = 0;
        if (bin >= (int)bins.size())
            bin = bins.size()-1;
        return bins[bin]/(float)totalEntries;
    }

    void Insert(const float value) {
        int bin = (int)(bins.size()*(value - minValue)/(maxValue-minValue));
        if (bin >= 0 && bin < (int)bins.size()) {
            validEntries++;
            bins[bin]++;
        }
        totalEntries++;
    }

    void Cumulate() {
        for (unsigned int i=1; i < bins.size(); i++)
            bins[i] += bins[i-1];
    }

    float Avg() {
        float sum = 0.0f;
        for (unsigned int i=0; i < bins.size(); i++)
            sum += bins[i] * (i*(maxValue-minValue)/bins.size() + minValue);
        return sum/validEntries;
    }

    void Print(std::ostream &s) {
        for (unsigned int i=0; i < bins.size(); i++)
            s << i << " " << bins[i] << std::endl;
    }

    void PrintNormalized() {
        std::cout << "entries = " << totalEntries << ", valid entries = " << validEntries << std::endl;
        for (unsigned int i=0; i < bins.size(); i++)
            std::cout << bins[i]/(double)totalEntries << std::endl;
    }
};

struct DirectionBox {
    int faceIndex;
    // range of directions
    float minU;
    float minV;
    float maxU;
    float maxV;

    DirectionBox():faceIndex(0),minU(-1.0f),minV(-1.0f),maxU(1.0f),maxV(1.0f) {}

    float GetVolume() const {
        return (maxU - minU)*(maxV - minV);
    }

    int GetMajorAxis() const { return faceIndex%3; }

    Vector3f GetMajorDirection() const {
        Vector3f res(0,0,0);
        res [GetMajorAxis()] = MajorPositive() ? 1.0f : -1.0f;
        return res;
    }

    bool MajorPositive() const {
        return (faceIndex<3);
    }

    int GetAxisU() const {
        return (GetMajorAxis() + 1)%3;
    }

    int GetAxisV() const {
        return (GetMajorAxis() + 2)%3;
    }

    Vector3f GetNormalMinU() const {
        int major = GetMajorAxis();
        int minor = GetAxisU();

        Vector3f v(0,0,0);
        v[major] = MajorPositive() ? minU : -minU;
        v[minor] = MajorPositive() ? -1.0f : 1.0f;
        return Normalize(v);
    }

    Vector3f GetNormalMaxU() const {
        int major = GetMajorAxis();
        int minor = GetAxisU();

        Vector3f v(0,0,0);
        v[major] = MajorPositive() ? -maxU : maxU;
        v[minor] = MajorPositive() ? 1.0f : -1.0f;
        return Normalize(v);
    }

    Vector3f GetNormalMinV() const {
        int major = GetMajorAxis();
        int minor = GetAxisV();

        Vector3f v(0,0,0);
        v[major] = MajorPositive() ? minV : -minV;
        v[minor] = MajorPositive() ? -1.0f : 1.0f;
        return Normalize(v);
    }

    Vector3f GetNormalMaxV() const {
        int major = GetMajorAxis();
        int minor = GetAxisV();

        Vector3f v(0,0,0);
        v[major] = MajorPositive() ? -maxV : maxV;
        v[minor] = MajorPositive() ? 1.0f : -1.0f;
        return Normalize(v);
    }

    Vector3f UVToDirection(const float u, const float v) const {
        Vector3f dir;
        dir[GetMajorAxis()] = MajorPositive() ? 1.0f : -1.0f;
        dir[GetAxisU()] = MajorPositive() ? u : -u;
        dir[GetAxisV()] = MajorPositive() ? v : -v;

        return Normalize(dir);
    }
};

}  // namespace frustumshafts

class FrustumShaft
{
public:
    struct Plane {
        Vector3f normal;
        float D;
        // Cached topological information about the closest corner of an AABB
        // to the half-space bounded by this plane and containing the shaft
        unsigned char nearCornerIndices[3];
    };

    static const unsigned int maxCullingPlanesCount = 11;

    FrustumShaft(): cullingPlanesCount(0) {}
    FrustumShaft(const Bounds3f &box,
                 const frustumshafts::DirectionBox &dirBox):
                     cullingPlanesCount(0), spatialBox(box), directionBox(dirBox) {}

    void InitGeometry();
    void PreprocessCullingPlanes();
    bool overlaps(const Bounds3f& box) const;
    void GenerateSampleRays(std::vector<Ray> &rays,
                            FrustumShaftBuilder *collection = NULL) const;
    float GetHitProbability(const Bounds3f &box,
                            const std::vector<Ray> &rays,
                            FrustumShaftBuilder *collection) const;
    Ray GetCenterRay();
    void Print(std::ostream &s) {
        s << "box: " << spatialBox.pMin << " " << spatialBox.pMax << std::endl;
        s << "dirbox: f=" << directionBox.faceIndex <<
             " (" << directionBox.minU << "," << directionBox.minV << ")(" << directionBox.maxU << "," << directionBox.maxV << ")" << std::endl;
        s << "major=" << directionBox.GetMajorAxis() << " u=" << directionBox.GetAxisU() << " v=" << directionBox.GetAxisV() << std::endl;
    }

protected:
    // A tolerance level with which the geometric queries will be carried out
#ifdef _MSC_VER
    // compilation fix for MSVC 2013
    #define shaftEpsilon 1e-5f
#else
    static constexpr float shaftEpsilon = 1e-5f;
#endif

    unsigned int cullingPlanesCount;
    Plane cullingPlanes[maxCullingPlanesCount];

public:
    Bounds3f spatialBox;
    frustumshafts::DirectionBox directionBox;
};

class FrustumShaftBuilder
{
public:
    FrustumShaftsAccel* accel = nullptr;
    Bounds3f Bbox;
    Vector3i originResolution;

    // environment parameters

    // 5D structure resolution
    unsigned int numCells;
    unsigned int directionResolution;

    // build parameters
    float minProb;
    unsigned int sampleRays;
    unsigned int maxEntries;
    bool cullByGeometry;
    bool useRelativeMemory;

    // build acceleration/memory consumption control methods
    unsigned int minBVHNodes;
    bool useCLMap;
    bool onlyOccupiedCells;
    unsigned int usedShaftsPercentage;

    // shaft sampling values
    unsigned int xResolution;
    unsigned int yResolution;
    unsigned int pixelSamples;

    struct ThreadData
    {
        Bounds3f lastBox;
        bool overlaps;
        unsigned int uniqueSequences;
        unsigned int totalSequences;
    };

    std::vector<ThreadData> threadData;
    std::mutex buildMutex;
    std::unordered_map<std::string, int> CLMap;
    frustumshafts::Histogram CLHistogram;

    struct SortEntry {
        unsigned int count;
        unsigned int index;
        friend bool operator<(const SortEntry &a, const SortEntry &b) {
            return a.count > b.count;
        }
    };
    std::vector<SortEntry> raysInShafts;
    unsigned int shaftSubset;

    int uniqueSequences;
    int totalSequences;

    FrustumShaft IndexToShaft(const unsigned int index) {
        unsigned int i = index;
        unsigned int iz = i/(originResolution.y*originResolution.x*6*directionResolution*directionResolution);
        i -= iz*(originResolution.y*originResolution.x*6*directionResolution*directionResolution);
        unsigned int iy = i/(originResolution.x*6*directionResolution*directionResolution);
        i -= iy*(originResolution.x*6*directionResolution*directionResolution);
        unsigned int ix = i/(6*directionResolution*directionResolution);
        i -= ix*(6*directionResolution*directionResolution);
        unsigned int iface = i/(directionResolution*directionResolution);
        i -= iface*(directionResolution*directionResolution);
        unsigned int iv = i/directionResolution;
        i -= iv*directionResolution;
        unsigned int iu = i;

        Bounds3f spatBox;
        spatBox.pMin.x = Bbox.pMin.x + ix*(Bbox.pMax.x - Bbox.pMin.x)/originResolution.x;
        spatBox.pMax.x = Bbox.pMin.x + (ix+1)*(Bbox.pMax.x - Bbox.pMin.x)/originResolution.x;
        spatBox.pMin.y = Bbox.pMin.y + iy*(Bbox.pMax.y - Bbox.pMin.y)/originResolution.y;
        spatBox.pMax.y = Bbox.pMin.y + (iy+1)*(Bbox.pMax.y - Bbox.pMin.y)/originResolution.y;
        spatBox.pMin.z = Bbox.pMin.z + iz*(Bbox.pMax.z - Bbox.pMin.z)/originResolution.z;
        spatBox.pMax.z = Bbox.pMin.z + (iz+1)*(Bbox.pMax.z - Bbox.pMin.z)/originResolution.z;

        frustumshafts::DirectionBox dirBox;
        dirBox.faceIndex = iface;
        dirBox.minU = (2*(int)iu - (int)directionResolution)/(float)directionResolution;
        dirBox.maxU = (2*((int)iu+1) - (int)directionResolution)/(float)directionResolution;
        dirBox.minV = (2*(int)iv - (int)directionResolution)/(float)directionResolution;
        dirBox.maxV = (2*((int)iv+1) - (int)directionResolution)/(float)directionResolution;

        FrustumShaft shaft(spatBox, dirBox);
        shaft.InitGeometry();

        return shaft;
    }

    void Init(FrustumShaftsAccel* _accel, const ParamSet &ps);
    bool UseShafts() const {
        // Experimental threshold on #nodes of a BVH for which it pays off to build shafts
        if (accel->totalNodes < minBVHNodes)
            accel->shaftTraversal = false;
        if (!accel->shaftTraversal)
            std::cout << "Class \"" << accel->BVHname << "\" BVH too small (" << accel->totalNodes << " nodes), plain BVH acceleration under way..." << std::endl;
        return accel->shaftTraversal;
    }
    void Build(const bool isMainBVH);
    float CullBVHbyShaftDFS(FrustumShaft &shaft, std::vector<int> &candidateList);
    int CullNodeSubtree(const int id,
                        FrustumShaft &shaft,
                        BVHAccel::LinearBVHNode* bvhNodes,
                        int &opened,
                        int &culled,
                        int depth = 0);
    Vector3i ComputeSpatialResolution(const Bounds3f &box,
                                      const int cells);
    bool OverlapsLeaf(const Bounds3f &box);
    void ComputeLUT(uint64_t index);
};

}  // namespace pbrt

#endif  // PBRT_ACCELERATORS_FRUSTUM_SHAFTS_H
