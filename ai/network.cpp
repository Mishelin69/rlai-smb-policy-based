#include "./network.hpp"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include <iostream>
#include <new>

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

    this->current_frame += 1;

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
        this->current_frame = 0;
    }

    //do a passtrough on a the current state and move data to CPU
    //of course, using the newest actor and conv network
    //-1 because this thing is 1 into the future sort of
    passtrough(this->actor, this->conv, this->cuda_activations, current_frame-1);

    //this basically just performs softmax over newest predictions we recorded
    //memsets the softmax to gpu and returns picked option, the one with highest
    //probability calculated by softmax
    return this->pick_action();
}

uint32_t RLAgent::pick_action() {

    float rng_number = this->dist(this->gen);
    float cumulative_probability = 0.0f;

    for (size_t i = 0; i < this->actor_out; ++i) {

        cumulative_probability += this->cpu_final_predict_pass[i];

        if (cumulative_probability > rng_number) {
            return i;
        }
    }

    return actor_out - 1;
}

void RLAgent::passtrough(Actor actor, ConvNetwork conv, float* preds, uint32_t i) {

    //Okay Assign "pass method to every network in the architecture and make it work :thumbs_up:
    //for now a reference how it should look I guess ?? ?? ?? ???????
    float* cuda_frame = this->cuda_env + i*MAP_SIZE;

    GPU::Tensor input = GPU::Tensor { cuda_frame, CNN_L1_IN, CNN_L1_IN, CNN_L1_IN_DEPTH };
    GPU::Tensor output = GPU::Tensor { preds, CNN_L1_OUT, CNN_L1_OUT, CNN_L2_OUT_DEPTH };

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

    size_t cnn_weights = 

        //conv network first, and its layers
        GPU::mem_needed_align(sizeof(float) * 16 * 3 * 3 * 1, sizeof(float) * 4) + 
        GPU::mem_needed_align(sizeof(float) * 32 * 4 * 4 * 16, sizeof(float) * 4) + 
        //skip max pooling since nothing learnable there
        GPU::mem_needed_align(sizeof(float) * 32 * 3 * 3 * 32, sizeof(float) * 4) + 
        GPU::mem_needed_align(sizeof(float) * 128 * 64, sizeof(float) * 4);

    size_t critic_weights = 
        //critic network
        GPU::mem_needed_align(sizeof(float) * 64 * 64, sizeof(float) * 4) + 
        GPU::mem_needed_align(sizeof(float) * 64 * 1, sizeof(float) * 4);

    size_t actor_weights = GPU::mem_needed_align(sizeof(float) * 64 * 64, sizeof(float) * 4) + 
        GPU::mem_needed_align(sizeof(float) * 64 * 3, sizeof(float) * 4);

    this->weights_total = 
        //conv network first, and its layers
        cnn_weights + 
        //critic network
        critic_weights+ 
        //actor network
        actor_weights; 

    size_t cnn_biases = 
        GPU::mem_needed_align(sizeof(float) * 
                //cnn
                (16 + 32 + 32 + 128), sizeof(float) * 4); 

    size_t critic_biases = 
        GPU::mem_needed_align(sizeof(float) * 
                //critic 
                64 + 1, sizeof(float) * 4);

    size_t actor_biases = 
        GPU::mem_needed_align(sizeof(float) * 
                //actor:wq
                64 + 3, sizeof(float) * 4);

    this->biases_total = 
        //cnn
        cnn_biases + 
        //critic
        critic_biases + 
        //actor
        actor_biases;

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
        this->activations_total * 2 * BATCH_SIZE + //cuda_actv_old + cuda_actv 
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

    //I thought about alignment and all the possible issues but the stuff is x*y* 16 or 32
    //so always divisible by 4
    //0x0 (relative to the app)
    size_t p_offset = 0;

    float* weights_for_old = big_block + p_offset;
    this->cuda_layer_weights_old = weights_for_old;

    p_offset += weights_total;

    float* weights_for_new = big_block + p_offset;
    this->cuda_layer_weights = weights_for_new;

    p_offset += weights_total;

    float* biases_for_old = big_block + p_offset;
    this->cuda_layer_biases_old = biases_for_old;

    p_offset += biases_total;

    float* biases_for_new = big_block + p_offset;
    this->cuda_layer_biases = biases_for_new;

    p_offset += biases_total;

    float* space_actv_old = big_block + p_offset;
    this->old_cuda_activations = space_actv_old;

    p_offset += BATCH_SIZE * activations_total;

    float* space_actv_new = big_block + p_offset;
    this->cuda_activations = space_actv_new;

    p_offset += BATCH_SIZE * activations_total;

    float* space_env = big_block + p_offset;
    this->cuda_env = space_env;
    this->env.cuda_env = this->cuda_env;

    p_offset += MAP_SIZE * BATCH_SIZE;

    float* adv = big_block + p_offset;
    this->cuda_advantage = adv;

    p_offset += BATCH_SIZE;

    float* val = big_block + p_offset;
    this->cuda_values = val;

    p_offset += BATCH_SIZE;

    float* ret = big_block + p_offset;
    this->cuda_returns = ret;

    p_offset += BATCH_SIZE;

    //I WILL UPLOAD BEFORE TRAINING FROM CPU OVER TO GPU
    float* rwrd = big_block + p_offset;
    this->cuda_rewards = rwrd;

    p_offset += BATCH_SIZE;

    float* grad_w_out = big_block + p_offset;
    this->cuda_gradients_with_respect_out = grad_w_out;

    p_offset += grad_with_out * BATCH_SIZE;

    float* cnn_gradient = big_block + p_offset;
    this->cuda_cnn_gradient = cnn_gradient;

    p_offset += cnn_gradient_size * BATCH_SIZE;

    float* critic_gradient = big_block + p_offset;
    this->cuda_critic_gradient = critic_gradient;

    p_offset += critic_gradient_size * BATCH_SIZE;

    float* actor_gradient = big_block + p_offset;
    this->cuda_actor_gradient = actor_gradient;

    p_offset += actor_gradient_size * BATCH_SIZE;

    //TIME TO INIT LAYERS
    //FIRST cNN
    
    //no offset because we already have var for that stuff
    size_t cuda_w_offset = 0;
    size_t cuda_b_offset = 0;

    this->conv.init_self(this->gpu, weights_for_new + cuda_w_offset, biases_for_new + cuda_b_offset);
    cuda_w_offset += cnn_weights;
    cuda_b_offset += cnn_biases;

    this->critic.init_self(this->gpu, weights_for_new + cuda_w_offset, biases_for_new + cuda_b_offset);
    cuda_w_offset += critic_weights;
    cuda_b_offset += critic_biases;

    this->actor.init_self(this->gpu, weights_for_new + cuda_w_offset, biases_for_new + cuda_b_offset);
    cuda_w_offset += actor_weights;
    cuda_b_offset += actor_biases;

    cuda_w_offset = 0;
    cuda_b_offset = 0;

    this->copy_conv.init_self(this->gpu, weights_for_old + cuda_w_offset, biases_for_old + cuda_b_offset);
    cuda_w_offset += cnn_weights;
    cuda_b_offset += cnn_biases;

    this->copy_critic.init_self(this->gpu, weights_for_old + cuda_w_offset, biases_for_old + cuda_b_offset);
    cuda_w_offset += critic_weights;
    cuda_b_offset += critic_biases;

    this->copy_actor.init_self(this->gpu, weights_for_old + cuda_w_offset, biases_for_old + cuda_b_offset);
    cuda_w_offset += actor_weights;
    cuda_b_offset += actor_biases;

    //CPU MEM
    float* cpu_final_pp = new (std::nothrow) float[3];

    if (!cpu_final_pp) {
        std::cerr << "RLAgent::RLAgent() | Failed to allocate memory on the cpu!" << std::endl;
    }

    this->cpu_final_predict_pass = cpu_final_pp;

    //Okay so everything may be initialized maybe? ?? please ?? ?? ?? ??? ?? ??
}

////////////////////////////////////////////////////////
//////////////////////CNN CODE//////////////////////////
////////////////////////////////////////////////////////

void ConvNetwork::init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b) {

    size_t weights_offset = 0;
    size_t bias_offset = 0;

    const size_t l1_depth = 1;
    const size_t l1_input = 13;
    const size_t l1_kernel = 3;
    const size_t l1_feature_maps = 16;

    const size_t l2_depth = 16;
    const size_t l2_input = 11;
    const size_t l2_kernel = 4;
    const size_t l2_feature_maps = 32;

    const size_t l3_depth = 32;
    const size_t l3_input = 8;
    const size_t l3_pool = 2;

    const size_t l4_depth = 32;
    const size_t l4_input = 4;
    const size_t l4_kernel = 3;
    const size_t l4_feature_maps = 32;

    const size_t l5_neurons = 64;
    const size_t l5_input = 2*2*32;

    //CONV LAYER
    l1_13x13_16x3x3.init_self(gpu, GPU::ActivationFunction::ReLU, 
            l1_feature_maps, l1_depth, l1_kernel, l1_input, 
            cuda_w + weights_offset, cuda_b + bias_offset
        ); 

    weights_offset += GPU::mem_needed_align(sizeof(float) * l1_feature_maps * l1_kernel * 
            l1_kernel * l1_depth, sizeof(float) * 4); 

    bias_offset += l1_feature_maps;

    //CONV LAYER
    l2_11x11_32x4x4.init_self(gpu, GPU::ActivationFunction::ReLU, 
            l2_feature_maps, l2_depth, l2_kernel, l2_input, 
            cuda_w + weights_offset, cuda_b + bias_offset
            ); 

    weights_offset += GPU::mem_needed_align(sizeof(float) * l2_feature_maps * l2_kernel * 
            l2_kernel * l2_depth, sizeof(float) * 4); 

    bias_offset += l2_feature_maps;

    //MAX POOLING
    l3_8x8_2x2.init_self(gpu, l3_pool, l3_depth, l3_input);

    //CONV LAYER
    l4_4x4_32x3x3.init_self(gpu, GPU::ActivationFunction::ReLU, 
            l4_feature_maps, l4_depth, l4_kernel, l4_input, 
            cuda_w + weights_offset, cuda_b + bias_offset
            ); 

    weights_offset += GPU::mem_needed_align(sizeof(float) * l4_feature_maps * l4_kernel * 
            l4_kernel * l4_depth, sizeof(float) * 4); 

    bias_offset += l4_feature_maps;

    l5_2x2x32_64.init_self(gpu, cuda_w + weights_offset, cuda_b + bias_offset, 
            l5_neurons, l5_input, GPU::ActivationFunction::ReLU, GPU::ActivationFunction::DerReLU);
}

void ConvNetwork::pass(float* state, float* out, cudaStream_t stream) {

    float* cuda_inp = state;
    float* cuda_out = out;

    GPU::Tensor input = GPU::Tensor { cuda_inp, CNN_L1_IN, CNN_L1_IN, CNN_L1_IN_DEPTH };
    GPU::Tensor output = GPU::Tensor { cuda_out, CNN_L1_OUT, CNN_L1_OUT, CNN_L2_OUT_DEPTH };

    //go with nullptr just to verify something :)
    l1_13x13_16x3x3.convolve(input, GPU::Tensor {
            nullptr, 3, 3, 16 }, output.dat_pointer, stream);

}

////////////////////////////////////////////////////////
////////////////////CRITIC CODE/////////////////////////
////////////////////////////////////////////////////////

void Critic::init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b) {

    size_t weights_offset = 0;
    size_t biases_offset = 0;

    size_t l1_neurons = 64;
    size_t l1_input = 64;

    size_t l2_neurons = 1;
    size_t l2_input = 64;

    l1_64_64.init_self(gpu, cuda_w + weights_offset, cuda_b + biases_offset, 
            l1_neurons, l1_input, GPU::ActivationFunction::ReLU, GPU::ActivationFunction::DerReLU);

    weights_offset += l1_input * l1_neurons;
    biases_offset += l1_neurons;

    l2_64_1.init_self(gpu, cuda_w + weights_offset, cuda_b + biases_offset, 
            l2_neurons, l2_input, GPU::ActivationFunction::ReLU, GPU::ActivationFunction::DerReLU);

}
////////////////////////////////////////////////////////
////////////////////ACTOR CODE//////////////////////////
////////////////////////////////////////////////////////

void Actor::init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b) {

    size_t weights_offset = 0;
    size_t biases_offset = 0;

    size_t l1_neurons = 64;
    size_t l1_input = 64;

    size_t l2_neurons = 3;
    size_t l2_input = 64;

    l1_64_64.init_self(gpu, cuda_w + weights_offset, cuda_b + biases_offset, 
            l1_neurons, l1_input, GPU::ActivationFunction::ReLU, GPU::ActivationFunction::DerReLU);

    weights_offset += l1_input * l1_neurons;
    biases_offset += l1_neurons;

    l2_64_3.init_self(gpu, cuda_w + weights_offset, cuda_b + biases_offset, 
            l2_neurons, l2_input, GPU::ActivationFunction::ReLU, GPU::ActivationFunction::DerReLU);

}
