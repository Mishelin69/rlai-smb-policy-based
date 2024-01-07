#include "./network.hpp"

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

        cumulative_probability += this->cuda_final_predict_pass[i];

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

    this->weights_total = 
        //conv network first, and its layers
        GPU::mem_needed_align(sizeof(float) * 16 * 3 * 3, sizeof(float) * 4) + 
        GPU::mem_needed_align(sizeof(float) * 32 * 4 * 4, sizeof(float) * 4) + 
        //skip max pooling since nothing learnable there
        GPU::mem_needed_align(sizeof(float) * 32 * 3 * 3, sizeof(float) * 4) + 
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

    this->actor_gradient_size = 
        GPU::mem_needed_align(sizeof(float) * 
            (, sizeof(float) * 4);

    size_t memory_to_allocate = 0 
        + 10;



    }
