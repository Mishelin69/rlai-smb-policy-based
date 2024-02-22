#pragma once
#include "./allocator/allocator.hpp"
#include "layers/convolution/conv_layer.hpp"
#include "layers/dense/dense_layer.hpp"
#include "layers/max_pooling/max_pooling.hpp"
#include "../data/environment/environment.hpp"
#include "../thread-pool/pool.hpp"
#include <random>

#define BATCH_SIZE 32
#define CUDA_STREAMS 4

#ifndef THREAD_POOL_SIZE
    #define THREAD_POOL_SIZE 16
#endif

#ifndef RLAGENT_EPSILON
    #define RLAGENT_EPSILON 0.2f
#endif

//discount factor
#ifndef RLAGENT_GAMMA
    #define RLAGENT_GAMMA 0.9f
#endif

#define AGENT_NUM_ACTIONS 3

const size_t CNN_L1_IN = 13;
const size_t CNN_L1_OUT = 11;
const size_t CNN_L1_IN_DEPTH = 1;
const size_t CNN_L1_OUT_DEPTH = 16;

const size_t CNN_L2_IN = 11;
const size_t CNN_L2_OUT = 8;
const size_t CNN_L2_IN_DEPTH = CNN_L1_OUT_DEPTH;
const size_t CNN_L2_OUT_DEPTH = 32;

const size_t CNN_L3_IN = 8;
const size_t CNN_L3_OUT = 4;
const size_t CNN_L3_IN_DEPTH = CNN_L2_OUT_DEPTH;
const size_t CNN_L3_OUT_DEPTH = 32;

const size_t CNN_L4_IN = 4;
const size_t CNN_L4_OUT = 2;
const size_t CNN_L4_IN_DEPTH = CNN_L3_OUT_DEPTH;
const size_t CNN_L4_OUT_DEPTH = 32;

const size_t CNN_L5_IN = 2;
const size_t CNN_L5_OUT = 64;
const size_t CNN_L5_IN_DEPTH = CNN_L4_OUT_DEPTH;
const size_t CNN_L5_OUT_DEPTH = 1;
//----------------================----------------
const size_t CRITIC_L1_IN = 64;
const size_t CRITIC_L1_OUT = 64;
const size_t CRITIC_L1_IN_DEPTH = CNN_L5_OUT_DEPTH;
const size_t CRITIC_L1_OUT_DEPTH = 1;

const size_t CRITIC_L2_IN = 64;
const size_t CRITIC_L2_OUT = 1;
const size_t CRITIC_L2_IN_DEPTH = CRITIC_L1_OUT_DEPTH;
const size_t CRITIC_L2_OUT_DEPTH = 1;
//----------------================----------------
const size_t ACTOR_L1_IN = 64;
const size_t ACTOR_L1_OUT = 64;
const size_t ACTOR_L1_IN_DEPTH = CNN_L5_OUT_DEPTH;
const size_t ACTOR_L1_OUT_DEPTH = 1;

const size_t ACTOR_L2_IN = 64;
const size_t ACTOR_L2_OUT = AGENT_NUM_ACTIONS;
const size_t ACTOR_L2_IN_DEPTH = ACTOR_L1_OUT_DEPTH;
const size_t ACTOR_L2_OUT_DEPTH = 1;

#include <cstdint>

class Actor {

private:
    //Possible outputs => 3
    //do softmax on CPU
    DenseLayer l1_64_64;
    DenseLayer l2_64_4;

    size_t gradient_size;
    size_t input_size;
    size_t output_size;

public:

    void act(float* cnn_processed, float* out, cudaStream_t stream);

    Actor();
    ~Actor();
    Actor(const Actor& other);
    Actor(const Actor&& other);

    //Accepts the summed/averaged surrogate function objective
    //from the network and then calculates the gradients based (based like me) 
    //on that information
    void calculate_gradient(float error_sur, float* out);

    //prob will do SGM maybe Adam idk we'll see but this 
    //shouldn't be an issue
    void apply_gradient(float* gradients, float loss);

    void init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b);
};

class Critic {

private:
    //Return Value for given state
    DenseLayer l1_64_64;
    DenseLayer l2_64_1;

    size_t gradient_size;
    size_t input_size;
    size_t output_size;

public:

    void value(float* cnn_processed, float* out, cudaStream_t stream);

    Critic();
    ~Critic();
    Critic(const Critic& other);
    Critic(const Critic&& other);

    //first we calculate the loss
    //lt = (Vt - Rt)^2
    //sum them up and use the scalar to calulate the gradient with
    //comment: why so fancy calling it a "scalar" lol, what was I thinking :D
    //respect to the loss
    void calculate_gradient(float* values, float* returns, float* out);

    //prob will do SGM maybe Adam idk we'll see but this 
    //shouldn't be an issue
    void apply_gradient(float* gradients, float loss);

    void init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b);
};

//include conv_layer_biases, max pooling, first without and then with
class ConvNetwork {

private:

    //soon to come layer definitions and stuff, will be implemented
    //not now tho a bit too lazy rn
    //not lazy anymore 0:
    ConvolutionalLayer l1_13x13_16x3x3; //ReLU activation
    ConvolutionalLayer l2_11x11_32x4x4; //ReLU activation
    //DONT FORGET TO RESET DELTA STUFF PLS FOR MAX POOLING TO 0 :sob:
    MaxPooling l3_8x8_2x2;
    ConvolutionalLayer l4_4x4_32x3x3; //ReLU activation
    DenseLayer l5_2x2x32_64;

    size_t gradient_size;
    size_t input_size;
    size_t output_size;

public:

    void pass(float* state, float* out, cudaStream_t stream);

    ConvNetwork();
    ~ConvNetwork();
    ConvNetwork(ConvNetwork& other) = default;
    ConvNetwork(const ConvNetwork&& other);

    void init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b);

    //this will be calculated two times each training steps 
    //due to Actor and Critic shared architecture
    //once done with Actors final gradients, then with Critics
    void calculate_gradient(float* input_gradients, float* out);

    //prob will do SGM maybe Adam idk we'll see but this 
    //shouldn't be an issue
    void apply_gradient(float* gradients, float loss);
};

class RLAgent {

    cudaStream_t streams[CUDA_STREAMS];

    Actor actor;
    Critic critic;
    ConvNetwork conv;                                    

    //Copies used for training
    Actor copy_actor;
    Critic copy_critic;
    ConvNetwork copy_conv;

    const uint32_t actor_layers;
    const uint32_t actor_in;
    const uint32_t actor_out;

    const uint32_t critic_layers;
    const uint32_t critic_in;
    const uint32_t critic_out;

    const uint32_t conv_layers;
    const uint32_t conv_in;
    const uint32_t conv_out;

    ////////////////////////////////////////////////////////////////////////
    //NOTE: FOR MORE INFO ABOUT MEMORY LAYOUT PLEASE LOOK INTO network.cpp//
    /////////ALL THE SPECIFICATIONS ARE IN THE RLAgent CONSTRUCTOR/////////
    ///////////////////////////////////////////////////////////////////////

    //space where all the weights reside in :) yay
    float* cuda_layer_weights;
    float* cuda_layer_weights_old;
    //same as above but for biases
    float* cuda_layer_biases;
    float* cuda_layer_biases_old;
    float* cuda_env = nullptr;

    //theta/theta_old
    //should be of size network activations * batch_size
    float* cuda_activations;
    float* old_cuda_activations;

    //copied over from cuda_activations, this is only for comfort
    //3 extra floats wont kill anybody
    //this will be on CPU
    float* cpu_final_predict_pass;

    float* cuda_rewards;
    float* cuda_advantage;
    float* cuda_values;
    float* cuda_returns;

    //this is not with respect to the output, this is 
    //just for weights and biases maybe idk, 
    //prob it'll be a separate thing for that
    //size of one times cuda streams
    float* cuda_actor_gradient;
    float* cuda_critic_gradient;
    float* cuda_cnn_gradient;

    //this is streams_size * max space needed for gradients for (var name too long to write D:)
    //two subsequent layers (yeah that much wasted memory :O)
    float* cuda_gradients_with_respect_out;

    uint32_t weights_total;
    uint32_t biases_total;
    uint32_t batch_size;
    size_t current_batch;
    size_t current_frame;
    size_t current_reward;

    size_t actor_gradient_size;
    size_t critic_gradient_size;
    size_t cnn_gradient_size;

    size_t activations_total;
    size_t grad_with_out;

    //For Stochastic Policy Action Selection
    std::random_device rd;
    std::mt19937 gen; 
    std::uniform_real_distribution<float> dist;


    GPU::Device gpu = GPU::Device(0);
    //we align for 128(4 * 32) bits since thats the most GPU warps can mem access at a time
    Allocator alloc = Allocator(this->gpu, sizeof(float)*4); 
    Environment env = Environment("../data/parse-mario-level-img/out", this->gpu, this->cuda_env);
    ThreadPool::Pool pool = ThreadPool::Pool(CUDA_STREAMS);

    private:

    void passtrough(Actor actor, ConvNetwork conv, float* preds, uint32_t i);

    //performs Stochastic Policy Action Selection
    uint32_t pick_action();

    //calculates all the advantages, which is basically just
    //values (from critic) sub returns =>
    //                At = Vt - Rt
    void calculate_advantage(Critic critic, float* advn);

    //calculates all Rt (r=> reward, returns Rt = rt + discount(gamma)*Rt+1)
    void calculate_return(float* returns);

    void learn();

    public:

    RLAgent();
    ~RLAgent() = default;

    uint32_t predict(
            const uint32_t mario_x, const uint32_t mario_y,
            const uint32_t n_enemies, const uint32_t timer, 
            const uint32_t e1_x, const uint32_t e1_y,
            const uint32_t e2_x, const uint32_t e2_y,
            const uint32_t e3_x, const uint32_t e3_y,
            const uint32_t e4_x, const uint32_t e4_y,
            const uint32_t e5_x, const uint32_t e5_y,
            const bool is_alive
            );

};
