#include "./environment.hpp"
#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdlib.h>
#include <string>

#ifndef BATCH_SIZE
    #define BATCH_SIZE 32
#endif

//IMPORTANT NOTE:
//SO TILES ARE A MULTIPLE OF 16
//THE GIVEN POSITION SHOULD BE THE RESULT OF floor(actual_level_position / 16)
//THE INPUT IS ALREADY CALCULATED ASSUME ITS CORRECT
//IMPLEMENT: Mirroring, Uploading the data to GPU, value(reward) function
//Mirror should just copy locally and then upload to gpu

void Environment::set_value_in_stage(const Vector2 v, const float value) {

    const TileMapData& stage = this->stages[this->current_stage];

    const uint32_t tiles_x = stage.tiles_x;
    const uint32_t tiles_y = stage.tiles_y;

    float* dat_ptr = stage.dat;
    dat_ptr[tiles_x * (int)v.y + (int)v.x] = value;
}

void Environment::update_player() {
    const auto player_pos = this->player_pos;
    this->set_value_in_stage(player_pos, PLAYER_VALUE); 
}

void Environment::inverse_update_player() {
    const auto player_pos = this->player_pos;
    this->set_value_in_stage(player_pos, EMPTY_VALUE); 
}

void Environment::unsimulate_enemies() {

    for (size_t i = 0; i < enemies_on_screen; ++i) {

        const Vector2 enm = this->enemy_pos[i];
        this->set_value_in_stage(enm, EMPTY_VALUE); 
    }
}

void Environment::simulate_enemies() {

    for (size_t i = 0; i < enemies_on_screen; ++i) {

        const Vector2 enm = this->enemy_pos[i];
        this->set_value_in_stage(enm, ENEMY_VALUE);        
    }
}

namespace fs = std::filesystem;

//Make sure that every value is zeroed out just because it would be normally 
//and then just set the values on ticks that would happen in games
Environment::Environment(const char* path_to_tile_data, GPU::Device& gpu, float* env):
    gpu(gpu), player_pos(Vector2_ZERO), last_frame_player_pos(Vector2_ZERO), timer(0), last_timer(0),
    current_stage(0), enemies_on_screen(0), current_frame(0), map_mirror(2*MAP_SIZE), cuda_env(env), 
    player_dead(false), rewards(BATCH_SIZE), rindex(0) {

        fs::path path(path_to_tile_data);

        if (!fs::is_directory(path)) {
            std::cerr << "Environment::Environment(): Given path is not a directory! | " << path_to_tile_data << std::endl;
            exit(-1);
        }

        for (const auto& e : fs::directory_iterator(path)) {

            //Ugh so awful and ugly
            //std::cout << "kttypower" << std::endl;

            std::cout << "PATH: " << e << std::endl;

            const wchar_t* wide_path = e.path().c_str();
            std::wstring wstr(wide_path);
            std::string wstr_to_str(wstr.begin(), wstr.end());
            const char* path = wstr_to_str.c_str();

            this->stages.push_back(TileMapData(path));
            //std::cout << " ImSkirby" << std::endl;
        }


        for (size_t i = 0; i < MAX_ENEMIES_ON_SCREEN; ++i) {
            enemy_pos[i] = Vector2_ZERO;
        }

        assert(this->map_mirror.capacity() == 2 * MAP_SIZE);

        for (size_t i = 0; i < 2 * MAP_SIZE; ++i) {
            this->map_mirror[i] = 0.0f;
        }

        this->current_mirror = 1;
    }

uint32_t snap_to_closest_x(uint32_t x) {
    return x / 16; //yes integer division and yes no snapping for now :(
}

uint32_t snap_to_closest_y(uint32_t x) {
    return ceilf(x / 16.0); //yes integer division and yes no snapping for now :(
}

void Environment::update_frame(
        uint32_t mario_x, uint32_t mario_y,
        uint32_t n_enemies, uint32_t timer, 
        uint32_t e1_x, uint32_t e1_y,
        uint32_t e2_x, uint32_t e2_y,
        uint32_t e3_x, uint32_t e3_y,
        uint32_t e4_x, uint32_t e4_y,
        uint32_t e5_x, uint32_t e5_y) {

    size_t max_lvl_y = stages[current_stage].tiles_y;

    //std::cout << "Before change: " << mario_y << std::endl;
    mario_x = snap_to_closest_x(mario_x) + 2;
    mario_y = static_cast<int>(snap_to_closest_y(mario_y)) + 1;

    e1_x = snap_to_closest_x(e1_x) + 2;
    e1_y = static_cast<int>(snap_to_closest_y(e1_y));

    e2_x = snap_to_closest_x(e2_x) + 2;
    e2_y = static_cast<int>(snap_to_closest_y(e2_y));

    e3_x = snap_to_closest_x(e3_x) + 2;
    e3_y = static_cast<int>(snap_to_closest_y(e3_y));

    e4_x = snap_to_closest_x(e4_x) + 2;
    e4_y = static_cast<int>(snap_to_closest_y(e4_y));

    e5_x = snap_to_closest_x(e5_x) + 2;
    e5_y = static_cast<int>(snap_to_closest_y(e5_y));


    if (mario_x > this->stages[this->current_stage].tiles_x || 
            mario_y > this->stages[this->current_stage].tiles_y) {

        std::cerr << "Environment::update_frame | Invalid Mario position!!" << std::endl;
        exit(-1);
    }

    //std::cout << "level_y: " << stages[current_stage].tiles_y << std::endl;
    //std::cout << "Mario_y: " << mario_y << std::endl;

    //set current frame data as old and then update it
    this->mirror_current();

    //update Marios position 

    this->inverse_update_player();

    //first ever frame
    if (this->last_frame_player_pos == Vector2_ZERO) {

        player_pos.x = mario_x;
        player_pos.y = mario_y;

        last_frame_player_pos = player_pos;

        this->last_timer = timer;
        this->timer = timer;

    } else {

        last_frame_player_pos = player_pos;

        player_pos.x = mario_x;
        player_pos.y = mario_y;

        this->last_timer = this->timer;
        this->timer = timer;
    }


    //call this to reset enemy position in the world
    this->unsimulate_enemies();

    this->enemies_on_screen = n_enemies;

    //update enemy positions
    //this should cascade down, in theory it will
    //no I didnt forget to add break, itll just be too redudant
    //and this is more elegant
    switch (enemies_on_screen) {

        case MAX_ENEMIES_ON_SCREEN:

            this->enemy_pos[MAX_ENEMIES_ON_SCREEN-1].x = e5_x;
            this->enemy_pos[MAX_ENEMIES_ON_SCREEN-1].y = e5_y;

        case MAX_ENEMIES_ON_SCREEN-1:

            this->enemy_pos[MAX_ENEMIES_ON_SCREEN-2].x = e4_x;
            this->enemy_pos[MAX_ENEMIES_ON_SCREEN-2].y = e4_y;

        case MAX_ENEMIES_ON_SCREEN-2:

            this->enemy_pos[MAX_ENEMIES_ON_SCREEN-3].x = e3_x;
            this->enemy_pos[MAX_ENEMIES_ON_SCREEN-3].y = e3_y;

        case MAX_ENEMIES_ON_SCREEN-3:

            this->enemy_pos[MAX_ENEMIES_ON_SCREEN-4].x = e2_x;
            this->enemy_pos[MAX_ENEMIES_ON_SCREEN-4].y = e2_y;

        case MAX_ENEMIES_ON_SCREEN-4:

            this->enemy_pos[MAX_ENEMIES_ON_SCREEN-5].x = e1_x;
            this->enemy_pos[MAX_ENEMIES_ON_SCREEN-5].y = e1_y;

            break;

        case 0:
        default:
            break;
    }

    simulate_enemies();

    this->update_player();

    this->mirror();

    compute_reward();
}

uint32_t Environment::index_last_frame() {

    return this->current_mirror;
}

void Environment::index_last_frame_update() {
    this->current_mirror ^= 1;
}

void Environment::mirror_current() {
    this->index_last_frame_update();
}

void Environment::mirror() {

    uint32_t last = this->index_last_frame();
    uint32_t current = 0;

    //copy the current 13x13 relative to player
    //13*13 => mario is in the "middle"
    //two to the left of mario and 10 in front
    //two below and ten above

    //std::cout << "Player x: " << this->player_pos.x << " y: " << player_pos.y << std::endl;
    const uint32_t top_left_x = this->player_pos.x - 2;
    const uint32_t top_left_y = this->player_pos.y - 10;

    std::cout << "Top left x: " << top_left_x << " Top left y: " << top_left_y << std::endl;

    const TileMapData& stage = this->stages[this->current_stage];
    float* dat_ptr = stage.dat;

    const uint32_t tiles_x = stage.tiles_x;

    //HANDLE PLAYER TOP LEFT OVERFLOW BY ADDING 0 TO IT 
    //OR GOING DOWN BUT PROB ADDING 0 BETTER

    for (size_t i = 0; i < MAP_DIM; ++i) {

        const uint32_t y = top_left_y + i;
        for (size_t j = 0; j < MAP_DIM; ++j) {
            //std::cout << " " << dat_ptr[y * tiles_x + j] << "\t";
            this->map_mirror[current*MAP_DIM + MAP_DIM*i + j] = dat_ptr[y * tiles_x + j];
        }

        //std::cout << std::endl;
    }

    //exit(-1);

    this->upload_to_gpu();
}

void Environment::upload_to_gpu() {

    const uint32_t cur_frame = 0;

    /*
    for (size_t i = 0; i < 13; ++i) {
        for (size_t j = 0; j < 13; ++j) {
           
            size_t index = cur_frame * MAP_SIZE + 13*i + j;
            std::cout << " " << this->map_mirror[index];
        }

        std::cout << "" << std::endl;
    }
    */

    //std::cout << "========================================" << std::endl;
    gpu.memcpy_host(
            this->map_mirror.data() + cur_frame * MAP_SIZE,
            this->cuda_env + this->rindex*MAP_SIZE,
            sizeof(float) * MAP_SIZE
            );

    /*
    gpu.print_mem(
            this->cuda_env + this->rindex*MAP_SIZE,
            MAP_SIZE
            );
            */

    this->current_frame += 1;
}

void Environment::player_died() {
    this->player_dead = true;
}

void Environment::compute_reward() {

    float v = this->player_pos.x - this->last_frame_player_pos.x;
    float c = this->last_timer - this->timer;
    float d = (this->player_dead) ? -15.0f : 0.f;

    //THIS ACCESS CAUSES CRASHES ???? WHY THO? ?? IDK
    this->rewards[this->rindex] = v + c + d;
    this->rindex += 1;
    this->player_dead = false;
}
