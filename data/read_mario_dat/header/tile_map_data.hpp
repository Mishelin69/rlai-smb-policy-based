#pragma once 

#include <memory>

#define TILE_DIMENSIONS 16
#define TILE_SIZE 16*16

#define MAX_PRECISION 3

class TileMapData {

public: 

    float* dat;
    uint32_t dat_size;

    uint32_t tiles_x;
    uint32_t tiles_y;

    uint32_t world;
    uint32_t level;

private: 

    static float* load_data(
            FILE* fs, float* out, uint32_t dat_size) noexcept;

public:

    TileMapData(const char* path);
    ~TileMapData() = default;
    TileMapData(TileMapData& other) = default;

};
