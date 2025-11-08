#ifndef _DARTLIB_BASIC_H_
#define _DARTLIB_BASIC_H_

#include <algorithm>
#include <vector>
#include <string>

namespace DartLib
{
// TODO: merge EdgeMapKey and FaceMapKey using templete
struct EdgeMapKey
{
    int indices[2];

    EdgeMapKey(const int* indices)
    {
        this->indices[0] = indices[0];
        this->indices[1] = indices[1];
        std::sort(std::begin(this->indices), std::end(this->indices));
    }

    bool operator==(const EdgeMapKey& other) const 
    { 
        return this->indices[0] == other.indices[0] &&
               this->indices[1] == other.indices[1]; 
    }
};

struct EdgeMapKey_hasher
{
    size_t operator()(EdgeMapKey const& key) const
    {
        // TODO: the number of the vertices depends on the design
        //       method of computing hash value
        //       if we consider to use double instead of size_t,
        //       then the range of hash value could be larger.
        size_t hash = (size_t) key.indices[0] * 1000000000L +
                      (size_t) key.indices[1];
        return hash;
    }
};

struct FaceMapKey
{
    std::vector<int> indices;

    FaceMapKey(const int* indices)
    {
        this->indices.resize(3);
        this->indices[0] = indices[0];
        this->indices[1] = indices[1];
        this->indices[2] = indices[2];
        std::sort(std::begin(this->indices), std::end(this->indices));
    }

    bool operator==(const FaceMapKey& other) const
    {
        return this->indices[0] == other.indices[0] &&
               this->indices[1] == other.indices[1] && 
               this->indices[2] == other.indices[2];
    }
};

struct FaceMapKey_hasher
{
    size_t operator()(FaceMapKey const& key) const
    {
        // TODO: the number of the vertices depends on the design
        //       method of computing hash value
        size_t hash = (size_t) key.indices[0] * 1000000000000L +
                      (size_t) key.indices[1] * 1000000L +
                      (size_t) key.indices[2];
        return hash;
    }
};

struct TetMapKey
{
    std::vector<int> indices;

    TetMapKey(const int* indices)
    {
        this->indices.resize(4);
        this->indices[0] = indices[0];
        this->indices[1] = indices[1];
        this->indices[2] = indices[2];
        this->indices[3] = indices[3];
        std::sort(std::begin(this->indices), std::end(this->indices));
    }

    bool operator==(const TetMapKey& other) const
    {
        return this->indices[0] == other.indices[0] &&
               this->indices[1] == other.indices[1] && 
               this->indices[2] == other.indices[2] &&
               this->indices[3] == other.indices[3];
    }
};

struct TetMapKey_hasher
{
    size_t operator()(TetMapKey const& key) const
    {
        size_t hash = (size_t) key.indices[0] * 100000000000000L + 
                      (size_t) key.indices[1] * 10000000000L + 
                      (size_t) key.indices[2] * 1000000L + 
                      (size_t) key.indices[3];
        return hash;
    }
};

template<int N>
struct TCell_MapKey
{
    std::vector<int> indices;

    TCell_MapKey(const int* indices)
    {
        this->indices.resize(N + 1);
        for (int i = 0; i < N + 1; ++i)
            this->indices[i] = indices[i];
        std::sort(std::begin(this->indices), std::end(this->indices));
    }

    bool operator==(const TCell_MapKey& other) const
    {
        for (int i = 0; i < N + 1; ++i)
        {
            if (this->indices[i] != other.indices[i])
                return false;
        }
        return true;
    }
};

template <int N>
struct TCell_MapKey_hasher
{
    std::string operator()(TCell_MapKey<N> const& key) const
    {
        std::string hash;
        for (int i = 0; i < N + 1; ++i)
        {
            hash += std::to_string(key.indices[i]);
            if (i < N) hash += "-";
        }
        return hash;
    }
};
}
#endif