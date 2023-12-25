#pragma once
#include "../read_mario_dat/header/tile_map_data.hpp"
#include "../../gpu/gpu.h"

#define MAX_ENEMIES_ON_SCREEN 5

struct Vector3 {
    float x;
    float y;
    float z;
};

class Environment {

private:

    TileMapData* stages;
    size_t current_stage;
    const char* dir_name;

    GPU::Device& gpu;

    float map_mirror[2];
    uint8_t current_frame;

    Vector3 player_pos;
    Vector3 last_frame_player_pos;

    size_t timer;

    Vector3 enemy_pos[MAX_ENEMIES_ON_SCREEN];
    size_t enemies_on_screen;

private:

    //returns the index, the index that is the last frame
    uint8_t index_last_frame();

public:

    Environment(const char* path_to_tile_data);
    ~Environment();
    Environment(Environment& other) = default;

    float compute_reward();

    void update_frame(Vector3 player_pos, size_t current_time,
            Vector3 enemy_pos[MAX_ENEMIES_ON_SCREEN]);

    void change_stage();
};
