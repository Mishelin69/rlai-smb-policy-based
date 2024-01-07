#pragma once
#include "../read_mario_dat/header/tile_map_data.hpp"
#include "../../gpu/gpu.h"
#include <vector>

#define MAX_ENEMIES_ON_SCREEN 5

#ifndef TILE_SIZE
    #define TILE_SIZE 16*16
#endif

//set define config
#ifndef MAP_DIM
    #define MAP_DIM 13
#endif

#define MAP_SIZE MAP_DIM*MAP_DIM

//set default config
#ifndef ENEMY_VALUE
    #define ENEMY_VALUE 0.2f
#endif

#ifndef PLAYER_VALUE
    #define PLAYER_VALUE 0.1f
#endif

#ifndef EMPTY_VALUE
    #define EMPTY_VALUE 0.0f
#endif

struct Vector2 {
    float x;
    float y;

    bool operator==(const Vector2& other) const {
        return this->x == other.x && this->y == other.y;
    }
};

const Vector2 Vector2_ZERO {0, 0};

class Environment {

private:

    std::vector<TileMapData> stages;
    size_t current_stage;

    GPU::Device& gpu;
    uint32_t current_frame;

    std::vector<float> map_mirror;
    uint32_t current_mirror = 1;

    Vector2 player_pos;
    Vector2 last_frame_player_pos;

    size_t timer;
    size_t last_timer;

    Vector2 enemy_pos[MAX_ENEMIES_ON_SCREEN];
    size_t enemies_on_screen;

    //store calculated rewards
    std::vector<float> rewards;
    size_t rindex;
    bool player_dead;

public:
    float* cuda_env;

private:

    //returns the index, the index that is the last frame
    uint32_t index_last_frame();
    void index_last_frame_update();

    void simulate_enemies();
    void unsimulate_enemies();
    
    void inverse_update_player();
    void update_player();

    void set_value_in_stage(const Vector2 v, const float value);

    void upload_to_gpu();

    //mirror current to old frame
    void mirror_current();

    //sets the new current frame
    void mirror();

public:

    Environment(const char* path_to_tile_data, GPU::Device& gpu, float* env);
    ~Environment();
    Environment(Environment& other) = default;

    //This funcitons tries to get player as right as possible 
    //so positive will be if player moved right, 0 if hasnt moved and left if
    //actor moved left, if the actor managed to die, -15, alway will get 
    //positive reward for surviving longer reward clipped between +-15
    void compute_reward();

    void player_died();

    void update_frame(
            const uint32_t mario_x, const uint32_t mario_y,
            const uint32_t n_enemies, const uint32_t timer, 
            const uint32_t e1_x, const uint32_t e1_y,
            const uint32_t e2_x, const uint32_t e2_y,
            const uint32_t e3_x, const uint32_t e3_y,
            const uint32_t e4_x, const uint32_t e4_y,
            const uint32_t e5_x, const uint32_t e5_y
            );

    void change_stage();
};
