
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

#include "frustumshafts.h"
#include "interaction.h"
#include "paramset.h"
#include "stats.h"
#include "parallel.h"
#include "progressreporter.h"

#include <algorithm>
#include <fstream>
#include <queue>
#include <stack>
#include <map>

namespace pbrt {

#if FRUSTUM_SHAFTS_TRAVERSAL_STATS
STAT_COUNTER("Shafts/Regular traversal steps", regularTraversalSteps);
STAT_COUNTER("Shafts/Shadow traversal steps", shadowTraversalSteps);
#endif
STAT_MEMORY_COUNTER("Memory/Shafts", shaftMemory);

namespace frustumshafts {

template<int T>
struct HaltonGenerator {
    HaltonGenerator(const bool initializeBases = false) {
        Reset();
        if (initializeBases) {
            for (int i=0; i < T; i++) {
                int base = FindNthPrime(i+1);
                if (base == 1)
                    base++;
                _invBases[i] = 1.0f/base;
            }
        }
    }

    void Reset() {
        for (int i=0; i < T; i++)
            prev[i] = 0;
    }

    void GetNext(float *a) {
        for (int i=0; i < T; i++) {
            a[i] = Halton(_invBases[i], prev[i]);
            prev[i] = a[i];
        }
    }

private:
    inline bool IsPrime(const int number) {
        bool isIt = true;
        for (int i = 2; i < number; i++) {
            if (number % i == 0) {
                isIt = false;
                break;
            }
        }
        if (number == 2)
            isIt = false;
        return isIt;
    }

    inline int FindNthPrime(const int index) {
        const int primes[] = {-1, 1, 3, 5, 7, 11, 13};
        if (index <= 6)
            return primes[index];

        int prime = 1;
        int found = 1;
        while (found != index) {
            prime += 2;
            if (IsPrime(prime) == true)
                found++;
        }
        return prime;
    }

    inline float Halton(float baseRec, float prev) {
        float r = 1.0f - prev;
        if (baseRec < r)
            return prev + baseRec;

        float h = baseRec;
        float hh;
        do {
            hh = h;
            h *= baseRec;
        } while (h > r);
        return prev + hh + h - 1.0f;
    }

    static float _invBases[T];
    float prev[T];
};

static HaltonGenerator<5> haltonInitializer(true);
template <> float HaltonGenerator<5>::_invBases[5] = {};

template<int T>
struct MersenneGenerator {
    MersenneGenerator():  // using the default_seed == 5489u; different values have fairly visible impact on the CL length
        generator(), dist(0.0f, 1.0f) {}

    void GetNext(float *a) {
        for (int i=0; i < T; i++)
            a[i] = dist(generator);
    }

private:
    std::mt19937 generator;
    std::uniform_real_distribution<float> dist;
};

template<typename T, const int N>
struct StaticStack {
    static const int capacity = N;
    T entries[N];
    int index;

    StaticStack<T, N>() {
        index = 0;
    }

    bool empty() { return index == 0; }
    T &top() { return entries[index-1]; }
    void pop() { index--; }
    void push(const T &a) {
        if (index < N)
            entries[index++] = a;
        else
            std::cerr << "stack overflow!" << std::endl;
    }
    void clear() { index = 0; }
    int size() { return index; }
};

}  // namespace frustumshafts

// FrustumShaftsAccel Local Declarations

// FrustumShaftsAccel Method Definitions
FrustumShaftsAccel::FrustumShaftsAccel(
        std::vector<std::shared_ptr<Primitive>> p,
        const ParamSet &ps,
        int maxPrimsInNode,
        SplitMethod splitMethod,
        int numCells,
        int directionResolution,
        bool shaftTraversal):
                BVHAccel(std::move(p), maxPrimsInNode, splitMethod),
                phase(Rendering),
                numCells(numCells),
                directionResolution(directionResolution),
                shaftTraversal(shaftTraversal) {
    ProfilePhase _(Prof::AccelConstruction);

    Bbox = WorldBound();
    // Inflate the Bbox a tiny bit to prevent numeric errors from causing troubles at the borders
    // of a tightly enclosed geometry (primitives lying precisely on the Bbox's surface)
    Bbox.Enlarge(Vector3f(1,1,1)*Bbox.Diagonal().Length()*1e-3f);

    InitLUT();

    builder = new FrustumShaftBuilder;
    builder->Init(this, ps);
}

void FrustumShaftsAccel::InitLUT()
{
    originResolution = ComputeSpatialResolution(Bbox, numCells);
    InitMultipliers();
    int m = GetMaxIndex();

    raysInShafts.resize(m);
    for (int i=0; i < m; i++)
        raysInShafts[i]=0;

    std::cout << "Frustum shaft max index = " << m << ", size of the lookup table = " << m*4/1000000 << " MB." << std::endl;
    std::cout << "Resolution: " <<
        originResolution.x << " x " <<
        originResolution.y << " x " <<
        originResolution.z << std::endl;
}

Vector3i FrustumShaftsAccel::ComputeSpatialResolution(const Bounds3f &box, const int cells)
{
    Vector3f s = box.Diagonal();
    float a = pow(s.x*s.y*s.z/cells, 1.0f/3.0f);

    int nx = std::max(1, (int)(s.x/a));
    int ny = std::max(1, (int)(s.y/a));
    int nz = std::max(1, (int)(s.z/a));
    return Vector3i(nx, ny, nz);
}

void FrustumShaftsAccel::BuildShafts(const bool isMainBVH)
{
    ProfilePhase _(Prof::FrustumShaftConstruction);
    builder->Build(isMainBVH);
    delete builder; builder = nullptr;
}

void FrustumShaftsAccel::InitCalibration()
{
    phase = Calibration;
}

void FrustumShaftsAccel::InitBuild()
{
    phase = Construction;
}

void FrustumShaftsAccel::InitRendering()
{
    phase = Rendering;
}

int FrustumShaftsAccel::GetMaxIndex() const {
    return originResolution.z * originResolution.y * originResolution.x * 6 *
           directionResolution * directionResolution;
}

void FrustumShaftsAccel::InitMultipliers() {
    spatialMult.x = originResolution.x / Bbox.Diagonal().x;
    spatialMult.y = originResolution.y / Bbox.Diagonal().y;
    spatialMult.z = originResolution.z / Bbox.Diagonal().z;
    indexMult[0] = directionResolution;
    indexMult[1] = directionResolution * directionResolution;
    indexMult[2] = 6 * directionResolution * directionResolution;
    indexMult[3] =
        originResolution.x * 6 * directionResolution * directionResolution;
    indexMult[4] = originResolution.y * originResolution.x * 6 *
                    directionResolution * directionResolution;
}

int FrustumShaftsAccel::RayToIndex(const Ray &ray) const {
    int ix = spatialMult.x * (ray.o.x - Bbox.pMin.x);
    if (ix < 0 || ix >= originResolution.x) return -1;

    int iy = spatialMult.y * (ray.o.y - Bbox.pMin.y);
    if (iy < 0 || iy >= originResolution.y) return -1;

    int iz = spatialMult.z * (ray.o.z - Bbox.pMin.z);
    if (iz < 0 || iz >= originResolution.z) return -1;

    int longestAxis;
    if (fabs(ray.d.x) > fabs(ray.d.y) && fabs(ray.d.x) > fabs(ray.d.z))
        longestAxis = 0;
    else if (fabs(ray.d.y) > fabs(ray.d.z))
        longestAxis = 1;
    else
        longestAxis = 2;

    int iu = (directionResolution * ray.d[(longestAxis + 1) % 3] / ray.d[longestAxis] +
              directionResolution) /
             2.0f;
    if (iu >= directionResolution) iu = directionResolution - 1;

    int iv = (directionResolution * ray.d[(longestAxis + 2) % 3] / ray.d[longestAxis] +
              directionResolution) /
             2.0f;
    if (iv >= directionResolution) iv = directionResolution - 1;

    int iface = longestAxis;
    if (ray.d[longestAxis] < 0) iface += 3;

    int index = iu + iv * indexMult[0] + iface * indexMult[1] +
                ix * indexMult[2] + iy * indexMult[3] + iz * indexMult[4];

    return index;
}

bool FrustumShaftsAccel::PopulateTraversalStack(const Ray &ray,
                                                int* nodesToVisit,
                                                int& toVisitOffset,
                                                int& currentNodeIndex) const {
    int index = RayToIndex(ray);
#if FIND_SHAFT_FOR_OUTLYING_RAYS
    // Ray origin might lie outside the Bbox: primary rays of camera outside the scene or any rays outside the instances
    if (index < 0) {
        // Find the closest intersection with the (scene or instance) Bbox and restart the ray from the corresponding voxel
        float tmin, tmax;
        const LinearBVHNode *node = &nodes[0];
        if (node->bounds.IntersectP(ray, &tmin, &tmax, true)) {
            Ray ray_tmp(ray.o + tmin*ray.d, ray.d);
            index = RayToIndex(ray_tmp);
        }
        else
            return false; // Scene/instance not hit
    }
#endif
    if (index >= 0) {
        switch (phase) {
            case Calibration:
            {
                raysInShafts[index]._a.fetch_add(1);
                break;
            }
            case Construction:
                break;
            case Rendering:
            {
                int data = shaftIndexToCLAddr[index];
                int s = candidateLists[data++];
                if (s > 0) {
                    for (int i = 0; i < s - 1; i++)
                        nodesToVisit[toVisitOffset++] = candidateLists[data++];
                    currentNodeIndex = candidateLists[data];
                }
                break;
            }
            default:
                break;
        }
        return true;
    }
    else {  // from-root traversal
        currentNodeIndex = 0;
        toVisitOffset = 0;
        return true;
    }
}

bool FrustumShaftsAccel::Intersect(const Ray &ray,
                                   SurfaceInteraction *isect) const {
    if (!nodes) return false;
    ProfilePhase p(Prof::AccelIntersect);
    bool hit = false;
    Vector3f invDir(1 / ray.d.x, 1 / ray.d.y, 1 / ray.d.z);
    int dirIsNeg[3] = {invDir.x < 0, invDir.y < 0, invDir.z < 0};
    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesToVisit[128];
    const LinearBVHNode *node;

    if (shaftTraversal) {
        if (!PopulateTraversalStack(ray, &nodesToVisit[0], toVisitOffset, currentNodeIndex))
            return false;
    }

    while (true) {
        node = &nodes[currentNodeIndex];
        // Check ray against BVH node
#if FRUSTUM_SHAFTS_TRAVERSAL_STATS
        if (phase == Rendering)
            ++regularTraversalSteps;
#endif
        if (node->bounds.IntersectP(ray, invDir, dirIsNeg)) {
            if (node->nPrimitives > 0) {
                // Intersect ray with primitives in leaf BVH node
                for (int i = 0; i < node->nPrimitives; ++i)
                    if (primitives[node->primitivesOffset + i]->Intersect(ray, isect))
                        hit = true;
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                // Put far BVH node on _nodesToVisit_ stack, advance to near
                // node
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }

    return hit;
}

bool FrustumShaftsAccel::IntersectP(const Ray &ray) const {
    if (!nodes) return false;
    ProfilePhase p(Prof::AccelIntersectP);
    Vector3f invDir(1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z);
    int dirIsNeg[3] = {invDir.x < 0, invDir.y < 0, invDir.z < 0};
    int nodesToVisit[128];
    int toVisitOffset = 0, currentNodeIndex = 0;
    const LinearBVHNode *node;

    if (shaftTraversal) {
        if (!PopulateTraversalStack(ray, &nodesToVisit[0], toVisitOffset, currentNodeIndex))
            return false;
    }

    while (true) {
        node = &nodes[currentNodeIndex];
#if FRUSTUM_SHAFTS_TRAVERSAL_STATS
        if (phase == Rendering)
            ++shadowTraversalSteps;
#endif
        if (node->bounds.IntersectP(ray, invDir, dirIsNeg)) {
            // Process BVH node _node_ for traversal
            if (node->nPrimitives > 0) {
                for (int i = 0; i < node->nPrimitives; ++i)
                    if (primitives[node->primitivesOffset + i]->IntersectP(ray)) {
                        return true;
                    }
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
    return false;
}

std::shared_ptr<FrustumShaftsAccel> CreateFrustumShaftsAccelerator(
    std::vector<std::shared_ptr<Primitive>> prims, const ParamSet &ps) {

    // Copied from BVHAccel
    std::string splitMethodName = ps.FindOneString("splitmethod", "sah");
    BVHAccel::SplitMethod splitMethod;
    if (splitMethodName == "sah")
        splitMethod = BVHAccel::SplitMethod::SAH;
    else if (splitMethodName == "hlbvh")
        splitMethod = BVHAccel::SplitMethod::HLBVH;
    else if (splitMethodName == "middle")
        splitMethod = BVHAccel::SplitMethod::Middle;
    else if (splitMethodName == "equal")
        splitMethod = BVHAccel::SplitMethod::EqualCounts;
    else {
        Warning("BVH split method \"%s\" unknown.  Using \"sah\".",
                splitMethodName.c_str());
        splitMethod = BVHAccel::SplitMethod::SAH;
    }
    int maxPrimsInNode = ps.FindOneInt("maxnodeprims", 4);

    int numCells = ps.FindOneInt("numCells", 0);
    int directionResolution = ps.FindOneInt("directionResolution", 0);
    bool shaftTraversal = ps.FindOneBool("shaftTraversal", false);

    auto fsa = std::make_shared<FrustumShaftsAccel>(std::move(prims), ps, maxPrimsInNode, splitMethod,
            numCells, directionResolution, shaftTraversal);
    return fsa;
}

void
FrustumShaft::InitGeometry()
{
    Vector3f normals[4];
    normals[0] = directionBox.GetNormalMinU();
    normals[1] = directionBox.GetNormalMinV();
    normals[2] = directionBox.GetNormalMaxU();
    normals[3] = directionBox.GetNormalMaxV();

    for (int i=0; i < 4; i++) {
        float maxD = -1e26f;
        for (int j=0; j < 8; j++) {
            float d = Dot(normals[i], (Vector3f)spatialBox.Corner(j));
            if (d > maxD)
                maxD = d;
        }
        cullingPlanes[i].normal = normals[i];
        cullingPlanes[i].D = -maxD;
    }

    cullingPlanes[4].normal = Vector3f(0,0,0);

    if (directionBox.MajorPositive()) {
        cullingPlanes[4].normal[directionBox.GetMajorAxis()] = -1.0f;
        cullingPlanes[4].D = spatialBox.pMin[directionBox.GetMajorAxis()];
    } else {
        cullingPlanes[4].normal[directionBox.GetMajorAxis()] = 1.0f;
        cullingPlanes[4].D = -spatialBox.pMax[directionBox.GetMajorAxis()];
    }

    cullingPlanesCount = 5;

    // Optionally add other up to two possible planes of the spatial bounding box

    if (directionBox.minU > 0) {
        cullingPlanes[cullingPlanesCount].normal = Vector3f(0,0,0);
        if (directionBox.MajorPositive()) {
            cullingPlanes[cullingPlanesCount].normal[directionBox.GetAxisU()] = -1.0f;
            cullingPlanes[cullingPlanesCount++].D = spatialBox.pMin[directionBox.GetAxisU()];
        } else {
            cullingPlanes[cullingPlanesCount].normal[directionBox.GetAxisU()] = 1.0f;
            cullingPlanes[cullingPlanesCount++].D = -spatialBox.pMax[directionBox.GetAxisU()];
        }
    }

    if (directionBox.maxU < 0) {
        cullingPlanes[cullingPlanesCount].normal = Vector3f(0,0,0);
        if (directionBox.MajorPositive()) {
            cullingPlanes[cullingPlanesCount].normal[directionBox.GetAxisU()] = 1.0f;
            cullingPlanes[cullingPlanesCount++].D = -spatialBox.pMax[directionBox.GetAxisU()];
        } else {
            cullingPlanes[cullingPlanesCount].normal[directionBox.GetAxisU()] = -1.0f;
            cullingPlanes[cullingPlanesCount++].D = spatialBox.pMin[directionBox.GetAxisU()];
        }
    }

    if (directionBox.minV > 0) {
        cullingPlanes[cullingPlanesCount].normal = Vector3f(0,0,0);
        if (directionBox.MajorPositive()) {
            cullingPlanes[cullingPlanesCount].normal[directionBox.GetAxisV()] = -1.0f;
            cullingPlanes[cullingPlanesCount++].D = spatialBox.pMin[directionBox.GetAxisV()];
        } else {
            cullingPlanes[cullingPlanesCount].normal[directionBox.GetAxisV()] = 1.0f;
            cullingPlanes[cullingPlanesCount++].D = -spatialBox.pMax[directionBox.GetAxisV()];
        }
    }

    if (directionBox.maxV < 0) {
        cullingPlanes[cullingPlanesCount].normal = Vector3f(0,0,0);
        if (directionBox.MajorPositive()) {
            cullingPlanes[cullingPlanesCount].normal[directionBox.GetAxisV()] = 1.0f;
            cullingPlanes[cullingPlanesCount++].D = -spatialBox.pMax[directionBox.GetAxisV()];
        } else {
            cullingPlanes[cullingPlanesCount].normal[directionBox.GetAxisV()] = -1.0f;
            cullingPlanes[cullingPlanesCount++].D = spatialBox.pMin[directionBox.GetAxisV()];
        }
    }

    for (int i=0; i < 4; i++) {
        Vector3f edge = Cross(cullingPlanes[(i+1)%4].normal, cullingPlanes[i].normal);
        if (directionBox.MajorPositive())
            edge = -edge;
        Vector3f nn = cullingPlanes[(i+1)%4].normal + cullingPlanes[i].normal;
        Vector3f tangent = Cross(edge, nn);
        Vector3f normal = Normalize(Cross(tangent, edge));

        float maxD = -1e26f;
        for (int j=0; j < 8; j++) {
            float d = Dot(normal, (Vector3f)spatialBox.Corner(j));
            if (d > maxD)
                maxD = d;
        }
        cullingPlanes[cullingPlanesCount].normal = normal;
        cullingPlanes[cullingPlanesCount++].D = -maxD;
    }

    PreprocessCullingPlanes();
}

void
FrustumShaft::PreprocessCullingPlanes()
{
    for (unsigned int i=0; i < cullingPlanesCount; i++) {
        cullingPlanes[i].nearCornerIndices[0] = (cullingPlanes[i].normal[0] > 0 ? 0 : 3);  // LO_X : HI_X
        cullingPlanes[i].nearCornerIndices[1] = (cullingPlanes[i].normal[1] > 0 ? 1 : 4);  // LO_Y : HI_Y
        cullingPlanes[i].nearCornerIndices[2] = (cullingPlanes[i].normal[2] > 0 ? 2 : 5);  // LO_Z : HI_Z
    }
}

void
FrustumShaft::GenerateSampleRays(std::vector<Ray> &rays,
                                 FrustumShaftBuilder *collection
                                ) const
{
    frustumshafts::MersenneGenerator<5> rng;
    //frustumshafts::HaltonGenerator<5> rng;
    for (unsigned int i=0; i < rays.size(); i++) {
        // get 5D random number and transform it into the 5D coordinates
        float rnd[5];
        rng.GetNext(rnd);
        Point3f origin = spatialBox.pMin + Vector3f(rnd[0], rnd[1], rnd[2])*(spatialBox.pMax - spatialBox.pMin);
        float u = directionBox.minU + rnd[3]*(directionBox.maxU - directionBox.minU);
        float v = directionBox.minV + rnd[4]*(directionBox.maxV - directionBox.minV);
        Vector3f direction = directionBox.UVToDirection(u, v);

        rays[i] = Ray(origin, direction);
    }

    // now shift the ray origins and compute tMax if required
    for (unsigned int i=0; i < rays.size(); i++) {
        float tmin = -Infinity, tmax = Infinity;  // set to whatever value to make compiler happy
        spatialBox.IntersectP(rays[i], &tmin, &tmax, false);  // we need the real (even negative) tNear here
        if (collection && collection->cullByGeometry) {
            float tshift = (tmax + 1e-3f);
            Ray ray(rays[i].o + rays[i].d*tshift, rays[i].d);
            SurfaceInteraction isect;
            if (collection->accel->Intersect(ray, &isect))
                rays[i].tMax = ray.tMax + (tshift - tmin);
        }
        // Shift the ray towards the entry point of the box
        rays[i].o = rays[i].o + rays[i].d*tmin;
    }
}

Ray
FrustumShaft::GetCenterRay()
{
    Point3f origin = spatialBox.Center();
    float u = directionBox.minU + 0.5f*(directionBox.maxU - directionBox.minU);
    float v = directionBox.minV + 0.5f*(directionBox.maxV - directionBox.minV);
    Vector3f direction = directionBox.UVToDirection(u, v);

    return Ray(origin, direction);
}

float
FrustumShaft::GetHitProbability(const Bounds3f &box,
                                const std::vector<Ray> &rays,
                                FrustumShaftBuilder *collection) const
{
    int hits = 0;
    for (unsigned int i=0; i < rays.size(); i++) {  // TODO: we could exit early once the result is known based on minProb in caller
        float tmin, tmax;
        if (box.IntersectP(rays[i], &tmin, &tmax, true)) {
            if (collection->cullByGeometry) {
                // Check whether the box intersection lies *before* the geometry intersection; if not, do not consider it a hit
                if (tmin > rays[i].tMax)
                    continue;
            }
            hits++;
        }
    }

    return hits/(float)rays.size();
}

bool
FrustumShaft::overlaps(const Bounds3f& box) const
{
    // Check against the explicit plane set
    for (unsigned int i = 0; i < cullingPlanesCount; i++) {
            Vector3f nearCorner(box.coord(cullingPlanes[i].nearCornerIndices[0]),
                                box.coord(cullingPlanes[i].nearCornerIndices[1]),
                                box.coord(cullingPlanes[i].nearCornerIndices[2]));
            float d = Dot(nearCorner, cullingPlanes[i].normal) + cullingPlanes[i].D;
            // The numerical error of d is approximately proportional to the normal length squared
            if (d > shaftEpsilon * Dot(cullingPlanes[i].normal, cullingPlanes[i].normal))
                return false;
    }
    return true;
}

void
FrustumShaftBuilder::Init(FrustumShaftsAccel* _accel, const ParamSet &ps)
{
    // Get parameters from the environment
    numCells = ps.FindOneInt("numCells", 0);
    directionResolution = ps.FindOneInt("directionResolution", 0);

    minProb = ps.FindOneFloat("minProb", 0.5f);
    sampleRays = ps.FindOneInt("sampleRays", 20);
    maxEntries = ps.FindOneInt("maxEntries", 31);
    cullByGeometry = ps.FindOneBool("cullByGeometry", true);
    useRelativeMemory = ps.FindOneBool("useRelativeMemory", false);

    minBVHNodes = ps.FindOneInt("minBVHNodes", 0);
    useCLMap = ps.FindOneBool("useCLMap", true);
    onlyOccupiedCells = ps.FindOneBool("onlyOccupiedCells", false);
    usedShaftsPercentage = ps.FindOneInt("usedShaftsPercentage", 100);
    xResolution = ps.FindOneInt("xresolution", 0);
    yResolution = ps.FindOneInt("yresolution", 0);
    pixelSamples = ps.FindOneInt("pixelsamples", 0);

    // Init own&other structures
    accel = _accel;
    Bbox = accel->Bbox;
    if (useRelativeMemory) {
        int dirs = 6 * directionResolution * directionResolution;
        int bvhBytes = accel->totalNodes * sizeof(BVHAccel::LinearBVHNode);
        int avgLength = 5;

        numCells = (5*numCells/1e6f) * bvhBytes / (dirs*sizeof(int)*avgLength*usedShaftsPercentage/100.0f);
        std::cout << "NumCells used = " << numCells << std::endl;
    }
    CLHistogram.Init(maxEntries+1, 0.0f, maxEntries+1);
    originResolution = accel->originResolution;
}

int
FrustumShaftBuilder::CullNodeSubtree(const int id,
                                     FrustumShaft &shaft,
                                     BVHAccel::LinearBVHNode* bvhNodes,
                                     int &opened,
                                     int &culled,
                                     int depth
                                    )
{
    int nodeId = id;
    if (depth > 2)
        return nodeId;

    while (true) {
        BVHAccel::LinearBVHNode &node = bvhNodes[nodeId];

        if (node.IsLeaf())
            return nodeId;

        BVHAccel::LinearBVHNode &left = bvhNodes[nodeId+1];
        BVHAccel::LinearBVHNode &right = bvhNodes[node.secondChildOffset];

        int nodes[2];
        int overlaps = 0;

        if (shaft.overlaps(left.bounds))
            nodes[overlaps++] = nodeId+1;

        if (shaft.overlaps(right.bounds))
            nodes[overlaps++] = node.secondChildOffset;

        if (overlaps == 0) {
            opened++;
            culled+=2;
            return -1;
        }

        if (overlaps > 1) {
            int nl = CullNodeSubtree(nodeId+1, shaft, bvhNodes, opened, culled, depth+1);
            int nr = CullNodeSubtree(node.secondChildOffset, shaft, bvhNodes, opened, culled, depth+1);
            if (nl == -1 && nr == -1)
                return -1;
            if (nl == -1)
                return nr;
            if (nr == -1)
                return nl;
            return nodeId;
        }

        opened++;
        culled++;

        // Descend to the overlapping node
        nodeId = nodes[0];
    }
    return nodeId;
}

float
FrustumShaftBuilder::CullBVHbyShaftDFS(FrustumShaft &shaft, std::vector<int> &candidateList)
{
    BVHAccel::LinearBVHNode* bvhNodes = accel->nodes;

    int opened = 0;
    int culled = 0;

    std::vector<Ray> rays;
    rays.resize(sampleRays);
    shaft.GenerateSampleRays(rays, this);

    std::stack<int> stack;

    // Center ray of the shaft is used for computing the node ordering
    Ray centerRay = shaft.GetCenterRay();

    float minProbLocal = minProb;

    float potential = 0.0f;
    const int maxTries = 10;
    for (int i=0; i < maxTries; i++) {
        candidateList.clear();
        opened = 0;
        culled = 0;
        stack.push(0);
        while (!stack.empty()) {
            int nodeIndex = stack.top();
            stack.pop();
            BVHAccel::LinearBVHNode &node = bvhNodes[nodeIndex];

            if (shaft.overlaps(node.bounds)) {
                if (node.IsLeaf()) {
                    candidateList.push_back(nodeIndex);
                } else {
                    float p = shaft.GetHitProbability(node.bounds, rays, this);
                    if (p >= minProbLocal) {  // Open the node
                        opened++;
                        if (!(centerRay.d[node.axis] > 0)) {
                            stack.push(node.secondChildOffset);
                            stack.push(nodeIndex+1);
                        } else {
                            stack.push(nodeIndex+1);
                            stack.push(node.secondChildOffset);
                        }
                    } else {
                        int n = CullNodeSubtree(nodeIndex, shaft, bvhNodes, opened, culled);
                        if (n >= 0) {
                            potential += (1.0f - p);
                            candidateList.push_back(n);
                        }
                    }
                }
            } else
                culled++;
        }
        if (candidateList.size() <= maxEntries)
            break;
        else
            minProbLocal += (i+1)/20.0f;  // Reduce the node list size in the next iteration
    }

    return potential;
}

bool
FrustumShaftBuilder::OverlapsLeaf(const Bounds3f &box)
{
    BVHAccel::LinearBVHNode* bvhNodes = accel->nodes;
    std::stack<int> stack;

    stack.push(0);
    while (!stack.empty()) {
        int nodeIndex = stack.top();
        stack.pop();
        BVHAccel::LinearBVHNode &node = bvhNodes[nodeIndex];
        if (Overlaps(node.bounds, box)) {
            if (!node.IsLeaf()) {
                stack.push(node.secondChildOffset);
                stack.push(nodeIndex+1);
            } else
                return true;
        }
    }
    return false;
}

void
FrustumShaftBuilder::Build(const bool isMainBVH)
{
    std::cout << "Class \"" << accel->BVHname << "\" BVH: " << accel->totalNodes << " nodes" << std::endl;

    raysInShafts.resize(accel->raysInShafts.size());
    int sum=0;
    for (unsigned int i=0; i < raysInShafts.size(); i++) {
        raysInShafts[i].count = accel->raysInShafts[i]._a;
        raysInShafts[i].index = i;
        sum += accel->raysInShafts[i]._a;
    }

    // Experimental threshold on #nodes of a BVH for which it pays off to build shafts
    if (accel->totalNodes < minBVHNodes) {
        std::cout << "... too small for building shafts" << std::endl;
        accel->shaftTraversal = false;
        return;
    }

    // Optionally find the most used shafts (only for the main BVH; in instance BVHs we don't run the calibration phase)
    if (isMainBVH) {
        shaftSubset = raysInShafts.size()*usedShaftsPercentage/100;
        if (shaftSubset != raysInShafts.size())
            std::nth_element(raysInShafts.begin(), raysInShafts.begin() + shaftSubset, raysInShafts.end());

        unsigned int subsetSum = 0;
        for (unsigned int i=0; i < shaftSubset; i++)
            subsetSum += raysInShafts[i].count;

        std::cout << "Rays = " << sum/1e6f << "M" << std::endl;
        if (shaftSubset != raysInShafts.size())
            std::cout << shaftSubset << " shafts form " << 100.0f*shaftSubset/raysInShafts.size() << "% of data,"
                      << " and contain " << (sum > 0 ? 100.0f*subsetSum/sum : 0) << "% of rays." << std::endl;
    }
    else {  // TODO: Each instance class should be configurable independently
        usedShaftsPercentage = 100;
        shaftSubset = raysInShafts.size();
#if USE_ONLY_OCCUPIED_CELLS_FOR_INSTANCES
        onlyOccupiedCells = true;  // Experimentally found to be a good trade-off between build vs render time
#else
        onlyOccupiedCells = false;
#endif
    }

    accel->shaftIndexToCLAddr.resize(raysInShafts.size());
    for (unsigned int i=0; i < accel->shaftIndexToCLAddr.size(); i++)
        accel->shaftIndexToCLAddr[i] = 0;

    // The default CL (consisting of the root node alone) for frustum shafts without their own CL
    accel->candidateLists.push_back(1);
    accel->candidateLists.push_back(0);

    totalSequences = 0;
    uniqueSequences = 0;

    threadData.resize(MaxThreadIndex());
    for (size_t i=0; i < threadData.size(); i++) {
        threadData[i].lastBox = Bounds3f();
        threadData[i].overlaps = true;
        threadData[i].uniqueSequences = 0;
        threadData[i].totalSequences = 0;
    }

    {   // block needed for timely RAII destruction of reporter before any other output occurs
        ProgressReporter reporter(shaftSubset, "Building shafts");
        // Better multithreaded performance with chunkSize being multiple of 6*d^2 (#consecutive shafts having the same base voxel)
        // with the OccupiedVoxels variant, which reuses computed voxel overlap info from previous shafts
        // Does not help much when shafts are shuffled, as with the ViewDependent variant (partially sorted most ray-hit subset),
        // or when the entire shaft collection is computed with the Complete variant (no voxel overlap info)
        ParallelFor([&](uint64_t index) { ComputeLUT(index); reporter.Update(); }, shaftSubset, 6*directionResolution*directionResolution);
        reporter.Done();
    }

    for (size_t i=0; i < threadData.size(); i++) {
        uniqueSequences += threadData[i].uniqueSequences;
        totalSequences += threadData[i].totalSequences;
    }

    CLMap.clear();
    raysInShafts.clear();
    accel->raysInShafts.clear();

    shaftMemory += accel->shaftIndexToCLAddr.size() * sizeof(accel->shaftIndexToCLAddr[0]);
    shaftMemory += accel->candidateLists.size() * sizeof(accel->candidateLists[0]);

#if FRUSTUM_SHAFTS_BUILD_STATS
    std::cout << "Candidate list length histogram:" << std::endl;
    CLHistogram.Print(std::cout);

    float avg = CLHistogram.Avg();
    std::cout << "Avg candidate list length = " << avg << std::endl;
#endif
}

void
FrustumShaftBuilder::ComputeLUT(uint64_t index)
{
    std::vector<int> candidateList;
    FrustumShaft shaft = IndexToShaft(raysInShafts[index].index);

    if (onlyOccupiedCells) {
        if (!(shaft.spatialBox == threadData[ThreadIndex].lastBox)) {
            threadData[ThreadIndex].lastBox = shaft.spatialBox;
            threadData[ThreadIndex].overlaps = OverlapsLeaf(shaft.spatialBox);
        }
        if (!threadData[ThreadIndex].overlaps)  // tests only the leaf nodes of BVH, not the referenced geometry primitives themselves
            return;
    }

    // Build the candidate list for this shaft
    CullBVHbyShaftDFS(shaft, candidateList);
    threadData[ThreadIndex].totalSequences++;

    {   // Store the candidate list
        std::lock_guard<std::mutex> lock(buildMutex);

        // Try to reduce the used memory by reusing identical CLs of different (usually adjacent) shafts
        if (useCLMap) {
            std::string key = "";
            for (unsigned int j=0; j < candidateList.size(); j++)
                key += std::to_string(candidateList[j]) + " ";

            auto mapIndex = CLMap.end();
            mapIndex = CLMap.find(key);

            // If not present yet, append the CL to the CL area; otherwise reuse the existing one
            if (mapIndex != CLMap.end()) {
                accel->shaftIndexToCLAddr[raysInShafts[index].index] = mapIndex->second;
                return;
            } else
                CLMap.emplace(key, accel->candidateLists.size());
        }
        threadData[ThreadIndex].uniqueSequences++;
        // Copy the nodes to the StackData and store index in LUT
        accel->shaftIndexToCLAddr[raysInShafts[index].index] = accel->candidateLists.size();
        accel->candidateLists.push_back(candidateList.size());
        for (unsigned int j=0; j < candidateList.size(); j++)
            accel->candidateLists.push_back(candidateList[j]);

        CLHistogram.Insert(candidateList.size());
    }
}

}  // namespace pbrt
