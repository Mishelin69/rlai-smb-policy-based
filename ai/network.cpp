#include "./network.hpp"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include <iostream>

uint32_t RLAgent::predict(const uint32_t mario_x, const uint32_t mario_y,
    const uint32_t n_enemies, const uint32_t timer,
    const uint32_t e1_x, const uint32_t e1_y,
    const uint32_t e2_x, const uint32_t e2_y,
    const uint32_t e3_x, const uint32_t e3_y,
    const uint32_t e4_x, const uint32_t e4_y,
    const uint32_t e5_x, const uint32_t e5_y, 
    const bool is_alive) {

    //updates environment and pushed data over to gpu so it can be utilized
    //by our network which is on there
    env.update_frame(mario_x, mario_y, n_enemies, timer, e1_x, e1_y, 
            e2_x,  e2_y, e3_x, e3_y, e4_x,  e4_y, e5_x,  e5_y);

    //we're one frame in behind with frames, since the very first state cant 
    //have immidiate reward because we first need to do something and then 
    //wait for response to see what happened (the very next update)
    if (current_frame > 1) {
        this->env.compute_reward();
        this->current_reward++;
    }

    //after updating the state to a new one and calculating reward
    //determine if it's time for training, then we can continue
    if (this->current_reward == this->batch_size) {
        learn();
        //DONT FORGET TO RESET ENV AFTER THIS, CURRENT REWARD ALSO
        this->current_reward = 0;
        this->env.rindex = 0;
    }

    //do a passtrough on a the current state and move data to CPU
    //of course, using the newest actor and conv network
    passtrough(this->actor, this->conv, this->cuda_activations);

    //this basically just performs softmax over newest predictions we recorded
    //memsets the softmax to gpu and returns picked option, the one with highest
    //probability calculated by softmax
    return this->pick_action();
}

uint32_t RLAgent::pick_action() {
    float rng_number = this->dist(this->gen);
    float cumulative_probability = 0.0f;

    for (size_t i = 0; i < this->conv_out; ++i) {

        cumulative_probability += this->cpu_final_predict_pass[i];

        if (cumulative_probability > rng_number) {
            return i;
        }
    }

    return actor_out - 1;
}

RLAgent::RLAgent():
    actor_layers(2), actor_in(128), actor_out(3),
    critic_layers(2), critic_in(128), critic_out(1), 
    conv_layers(5), conv_in(MAP_SIZE), conv_out(64) {

    for (size_t i = 0; i < CUDA_STREAMS; ++i) {

        auto ret = cudaStreamCreate(&(this->streams[i]));

        if (ret != cudaSuccess) {
            std::cerr << "RLAgent::RLAgent() | Couldn't allocate a cuda stream!" << std::endl;
        }
    }

    this->weights_total = 
        //conv network first, and its layers
        GPU::mem_needed_align(sizeof(float) * 16 * 3 * 3 * 1, sizeof(float) * 4) + 
        GPU::mem_needed_align(sizeof(float) * 32 * 4 * 4 * 16, sizeof(float) * 4) + 
        //skip max pooling since nothing learnable there
        GPU::mem_needed_align(sizeof(float) * 32 * 3 * 3 * 32, sizeof(float) * 4) + 
        GPU::mem_needed_align(sizeof(float) * 128 * 64, sizeof(float) * 4) + 
        //critic network
        GPU::mem_needed_align(sizeof(float) * 64 * 64, sizeof(float) * 4) + 
        GPU::mem_needed_align(sizeof(float) * 64 * 1, sizeof(float) * 4) + 
        //actor network
        GPU::mem_needed_align(sizeof(float) * 64 * 64, sizeof(float) * 4) + 
        GPU::mem_needed_align(sizeof(float) * 64 * 3, sizeof(float) * 4); 

    this->biases_total = 
        GPU::mem_needed_align(sizeof(float) * 
                //cnn
                (16 + 32 + 32 + 128 +
                 //critic 
                 64 + 1 + 
                 //actor:wq
                 64 + 3), sizeof(float) * 4);

    this->batch_size = 32;
    this->current_batch = 0;
    this->current_frame = 0;
    this->current_reward = 0;

    this->actor_gradient_size = GPU::mem_needed_align(
            sizeof(float) * (64*64 + 64*3 + 64 + 3), sizeof(float) * 4
            );

    this->critic_gradient_size = GPU::mem_needed_align(
            sizeof(float) * (64*64 + 64*1 + 64 + 3), sizeof(float) * 4
            );

    this->cnn_gradient_size = GPU::mem_needed_align(
            sizeof(float) * (144 + 16 + 8192 + 32 + 9216 + 32 + 8192 + 64), // Separating each component
            sizeof(float) * 4 // Alignment to 128 bits
            );

    this->activations_total = GPU::mem_needed_align(
            64 + 3 +
            64 + 1 +
            11 * 11 * 16 + 
            8 * 8 * 32 +
            4 * 4 * 32 +
            2 * 2 * 32,
            sizeof(float) * 4 // Alignment to 128 bits
            );

    this->grad_with_out = this->cnn_gradient_size = GPU::mem_needed_align(
            3 + 64 +
            1 + 64 + 
            64 + 2*2*32 + 4*4*32 + 8*8*32 + 11*11*16,
            //dense layer + conv layers
            sizeof(float) * 4 // Alignment to 128 bits
            );

    auto TOTAL_MEM_NEEDED = 
        this->weights_total * 2 + //weight needed for network and network old
        this->biases_total * 2 + //biases needed =||=
        this->activations_total * 2 * BATCH_SIZE + //cuda_actv + cuda_actv_old 
        MAP_SIZE * BATCH_SIZE + //env 
        BATCH_SIZE * 4 + //advantage, values, returns, rewards
        grad_with_out * BATCH_SIZE + 
        cnn_gradient_size * BATCH_SIZE +
        critic_gradient_size * BATCH_SIZE +
        actor_gradient_size * BATCH_SIZE; 

    int res = alloc.alloc_new_block(GPU::mem_needed_align(
                TOTAL_MEM_NEEDED
                ,sizeof(float) * 4 // Alignment to 128 bits

                //(Memory relative to the pointer)
                /*
                 * Memory organisation:
                 *  -- Space for the weights, old network will always start at 0x0
                 *  -- Space for biases, old network biases start at 0x0
                 *      >>The way I'll make this work is that after each update Ill copy over
                 *      >>stuff to that predifined location so when the network is initialized for the first time
                 *      >>with data, nothing has to change/copy/etc the values will get updatet automatically :)
                 *  -- Space for activations, always after network values and stuff
                 *      >>To describe what I've done above for me later down this rabbit hole
                 *      >>I allocate enough space for batch size because:
                 *          >>-All the states will be stored somewhere down(up?) the memory and 
                 *          >> I only need to store for batch size, because thats the max Ill need
                 *          >> (during training otherwise one will be enough)
                 *  -- Space for the environment
                 *      >> Self explanatory
                 *  -- Advantage(At), Values(Vt), Returns(Rt), Rewards(rt)
                 *      >>So the process is:
                 *          >>-Copy rewards from env stored on the cpu over to gpu
                 *          >>-Calculate Returns going backwards for more effeciency
                 *          >>-Have critic output Values for everything
                 *          >>-Calculate Advantage for every of the step we took
                 *  -- Space for gradient with respect to the output
                 *      >>Only store all the gradients with respect to the output over here
                 *      >>all for simplicity of the implementation, dont want to deal
                 *      >>with that much indexing and the performance hit
                 *      >>shouldnt be that big a of hit for me
                 *  -- Space for gradients with respect to the parameters of cNN, Critic, Actor
                 *      >>Basically space for gradients in the order specified above
                 *      >>Gradients for biases are accounted for so dw
                 */
        ));

    if (res == -1) {
        std::cerr << "RLAgent::RLAgent() | Memory allocation failed! Exiting app!!!!" << std::endl;
    }

    //Assigning pointers to their correct stuff
    //alloc the frm big blck
    float* big_block = alloc.alloc_space(TOTAL_MEM_NEEDED);

    if (!big_block) {
        std::cerr << "RLAgent::RLAgent() | Requesting memory failed! Exiting app!!!!" << std::endl;
    }

    //0x0 (relative to the app)
    float* weights_for_old = big_block;

}
