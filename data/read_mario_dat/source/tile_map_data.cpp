#include "../header/tile_map_data.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <stdio.h>
#include <vector>

//
// NOTE: WOULD USE `std::experimental::filesystem` but clangd won't stop whining
//       about it not existing so f this :) good old C way will do for now
//

struct NumberOffsetPair {
    uint32_t n;
    uint32_t off;
};

static NumberOffsetPair nocheck_strtouint32t(const char* c) noexcept {

    uint32_t n = 0;
    uint32_t i = 0;

    if (*c == '+') {
        ++i;
    }

    while (c[i]  >= '0' && c[i] <= '9') {

        n *= 10;
        n += c[i] - '0';

        ++i;
    }

    return NumberOffsetPair {n, i};
}

static std::vector<uint32_t> string_get_numbers(const char* c) noexcept {

    std::vector<uint32_t> v;

    uint32_t c_len = strlen(c);
    uint32_t i = 0;

    while (i < c_len) {

        if (c[i] >= '0' && c[i] <= '9') {
            
            NumberOffsetPair ret = nocheck_strtouint32t(c + i);

            v.push_back(ret.n);
            i += ret.off;

            continue;
        }

        ++i;
    }

    return v;
}

static void log_error(const char* t) noexcept {
    std::cerr << "Error: " <<  t << std::endl;
    exit(-1);
}

std::unique_ptr<float*> TileMapData::load_data(
        FILE* fs, std::unique_ptr<float*> out, uint32_t dat_size) noexcept {

    for (int i = 0; i < dat_size; ++i) {

        //first char is guaranteed to be there so valid to just do this
        char c = fgetc(fs);
        float x = c - '0';

        //just a whole number :)
        if (fgetc(fs) != '.') {
            (*out)[i] = x;
            continue;
        }

        //why +1 ? => just in case we reach the MAX_PRECISION there will
        //still be space there so for comfort we just check for it and in that 
        //case skip it because that would really mess up this program 
        //if we reached MAX_PRECISION and it would not be possible to
        //check for that extra space and this guarantees it :) 
        //sorry for this long ass essay I just wrote
        for (int j = 0; j < MAX_PRECISION + 1; ++j) {
            
            c = fgetc(fs);

            if (c == ' ') {
                break;
            }

            x += powf(10, -j) * (c - '0');
        }

        (*out)[i] = x;
    }

    return std::move(out);
}

TileMapData::TileMapData(const char* path) {

    if (!path || strlen(path) < 1) {
        log_error("Invalid path pointer!");
    }

    std::vector<uint32_t> level_info = string_get_numbers(path + 0);

    //This should only have 4 elems not more, not less
    uint32_t tiles_y = level_info[0];
    uint32_t tiles_x = level_info[1];

    uint32_t world = level_info[2];
    uint32_t level = level_info[3];

    uint32_t dat_size = tiles_y * tiles_x;

    //this shouldn't throw hopefully because that's the ugliest thing ever :)
    float* _dat = new (std::nothrow) float [dat_size];

    if (!_dat) {
        log_error("Couldn't allocate memory!");
    }

    FILE* fs = nullptr; 

    errno_t ret = fopen_s(&fs, path, "r");

    if (ret != 0) {
        std::cerr << "Error while opening a file! (" << ret << ")" << std::endl;
        exit(-1);
    }

    this->dat = std::make_unique<float *>(_dat);
    this->dat_size = dat_size;
    this->tiles_y = tiles_y;
    this->tiles_x = tiles_x;
    this->world = world;
    this->level = level;

    this->dat = TileMapData::load_data(fs, std::move(this->dat), this->dat_size);

    //TO BE CONTINUED
    //puk = pandos = rat = trash panda = big rat ??? 
}
