#pragma once
#include "./allocator/allocator.hpp"
#include "layers/convolution/conv_layer.hpp"
#include "layers/dense/dense_layer.hpp"
#include "layers/max_pooling/max_pooling.hpp"
#include "../data/environment/environment.hpp"
#include "../external/thread-pool/include/BS_thread_pool.hpp"
#include <random>

#define BATCH_SIZE 32
#define CUDA_STREAMS 4

#ifndef THREAD_POOL_SIZE
    #define THREAD_POOL_SIZE CUDA_STREAMS
#endif

#ifndef RLAGENT_EPSILON
    #define RLAGENT_EPSILON 0.2f
#endif

//discount factor
#ifndef RLAGENT_GAMMA
    #define RLAGENT_GAMMA 0.9f
#endif

#ifndef RLAGENT_LAMBDA
    #define RLAGENT_LAMBDA 0.9f
#endif

#define AGENT_NUM_ACTIONS 4

constexpr size_t CNN_L1_IN = 13;
constexpr size_t CNN_L1_OUT = 11;
constexpr size_t CNN_L1_IN_DEPTH = 1;
constexpr size_t CNN_L1_OUT_DEPTH = 16;

constexpr size_t CNN_L2_IN = 11;
constexpr size_t CNN_L2_OUT = 8;
constexpr size_t CNN_L2_IN_DEPTH = CNN_L1_OUT_DEPTH;
constexpr size_t CNN_L2_OUT_DEPTH = 32;

constexpr size_t CNN_L3_IN = 8;
constexpr size_t CNN_L3_OUT = 4;
constexpr size_t CNN_L3_IN_DEPTH = CNN_L2_OUT_DEPTH;
constexpr size_t CNN_L3_OUT_DEPTH = 32;

constexpr size_t CNN_L4_IN = 4;
constexpr size_t CNN_L4_OUT = 2;
constexpr size_t CNN_L4_IN_DEPTH = CNN_L3_OUT_DEPTH;
constexpr size_t CNN_L4_OUT_DEPTH = 32;

constexpr size_t CNN_L5_IN = 2;
constexpr size_t CNN_L5_OUT = 64;
constexpr size_t CNN_L5_IN_DEPTH = CNN_L4_OUT_DEPTH;
constexpr size_t CNN_L5_OUT_DEPTH = 1;
//----------------================----------------
constexpr size_t CRITIC_L1_IN = 64;
constexpr size_t CRITIC_L1_OUT = 64;
constexpr size_t CRITIC_L1_IN_DEPTH = CNN_L5_OUT_DEPTH;
constexpr size_t CRITIC_L1_OUT_DEPTH = 1;

constexpr size_t CRITIC_L2_IN = 64;
constexpr size_t CRITIC_L2_OUT = 1;
constexpr size_t CRITIC_L2_IN_DEPTH = CRITIC_L1_OUT_DEPTH;
constexpr size_t CRITIC_L2_OUT_DEPTH = 1;
//----------------================----------------
constexpr size_t ACTOR_L1_IN = 64;
constexpr size_t ACTOR_L1_OUT = 64;
constexpr size_t ACTOR_L1_IN_DEPTH = CNN_L5_OUT_DEPTH;
constexpr size_t ACTOR_L1_OUT_DEPTH = 1;

constexpr size_t ACTOR_L2_IN = 64;
constexpr size_t ACTOR_L2_OUT = AGENT_NUM_ACTIONS;
constexpr size_t ACTOR_L2_IN_DEPTH = ACTOR_L1_OUT_DEPTH;
constexpr size_t ACTOR_L2_OUT_DEPTH = 1;
//----------------================----------------
constexpr size_t cnn_activations_footprint = 
    CNN_L1_OUT*CNN_L1_OUT*CNN_L1_OUT_DEPTH + 
    CNN_L2_OUT*CNN_L2_OUT*CNN_L2_OUT_DEPTH + 
    CNN_L3_OUT*CNN_L3_OUT*CNN_L3_OUT_DEPTH + 
    CNN_L4_OUT*CNN_L4_OUT*CNN_L4_OUT_DEPTH + 
    CNN_L5_OUT;

constexpr size_t critic_activations_footprint = 
    CRITIC_L1_OUT + 
    CRITIC_L2_OUT;
constexpr size_t actor_activations_footprint = 
    ACTOR_L1_OUT + 
    ACTOR_L2_OUT;
//----------------================----------------
constexpr size_t critic_grad_l1 = CRITIC_L2_OUT;
constexpr size_t critic_grad_l2 = CRITIC_L1_OUT;

constexpr size_t actor_grad_l1 = CRITIC_L2_OUT;
constexpr size_t actor_grad_l2 = CRITIC_L1_OUT;

constexpr size_t cnn_grad_l1 = CNN_L5_OUT;
constexpr size_t cnn_grad_l2 = CNN_L4_OUT*CNN_L4_OUT*CNN_L4_OUT_DEPTH;
constexpr size_t cnn_grad_l3 = CNN_L3_OUT*CNN_L3_OUT*CNN_L3_OUT_DEPTH;
constexpr size_t cnn_grad_l4 = CNN_L2_OUT*CNN_L2_OUT*CNN_L2_OUT_DEPTH;
constexpr size_t cnn_grad_l5 = CNN_L1_OUT*CNN_L1_OUT*CNN_L1_OUT_DEPTH;

constexpr size_t critic_gradual_l1 = critic_grad_l1;
constexpr size_t critic_gradual_l2 = critic_gradual_l1 + critic_grad_l2;

constexpr size_t actor_gradual_l1 = actor_grad_l1;
constexpr size_t actor_gradual_l2 = actor_gradual_l1 + actor_grad_l2;

constexpr size_t cnn_gradual_l1 = cnn_grad_l1;
constexpr size_t cnn_gradual_l2 = cnn_gradual_l1 + cnn_grad_l2;
constexpr size_t cnn_gradual_l3 = cnn_gradual_l2 + cnn_grad_l3;
constexpr size_t cnn_gradual_l4 = cnn_gradual_l3 + cnn_grad_l4;
constexpr size_t cnn_gradual_l5 = cnn_gradual_l4 + cnn_grad_l5;
constexpr size_t network_gradual = cnn_gradual_l5 + actor_gradual_l2;

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
    friend class RLAgent;

public:

    void act(float* cnn_processed, float* out, cudaStream_t stream);

    Actor();
    ~Actor() = default;
    Actor(Actor& other) = default;
    Actor(const Actor&& other);

    //Accepts the summed/averaged surrogate function objective
    //from the network and then calculates the gradients based (based like me) 
    //on that information
    void calculate_gradient(float error_sur, float* out);

    //prob will do SGM maybe Adam idk we'll see but this 
    //shouldn't be an issue
    void apply_gradient(float* gradients, float loss);

    void init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b);
    void deep_copy(GPU::Device& gpu, const Actor& original);
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
    friend class RLAgent;

public:

    void value(float* cnn_processed, float* out, cudaStream_t stream);

    Critic();
    ~Critic() = default;
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

//include conv_layer_biases, max pooling, ncfirst without and then with
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
    friend class RLAgent;

public:

    void pass(float* state, float* out, cudaStream_t stream);

    ConvNetwork();
    ~ConvNetwork() = default;
    ConvNetwork(ConvNetwork& other) = default;
    ConvNetwork(const ConvNetwork&& other);

    void init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b);
    void deep_copy(GPU::Device& gpu, const ConvNetwork& original);

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
    float* cuda_activation_z;

    //copied over from cuda_activations, this is only for comfort
    //3 extra floats wont kill anybody
    //this will be on CPU
    float* cpu_final_predict_pass;
    float* copy_final_predict_pass;

    float* cuda_rewards;
    float* cuda_advantage;
    float* cuda_values;
    float* cuda_returns;
    float* cpu_rewards;

    float* old_cuda_action_taken;
    float* cuda_action_taken;
    float* cuda_all_prob_old;
    float* cuda_all_prob_cur;

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

    float* cuda_ppo_objective;
    float* cuda_discounted_rewards;
    float* cuda_gae_delta;
    float* cuda_gae;
    float* sum_ppo;
    float* sum_critic_loss;
    float* cuda_critic_mse;

    float* cuda_actor_grad_wrt_in;

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
    Environment env = Environment("C:/Users/milos/Desktop/home/coding/github/rlai-smb-policy-based/data/parse-mario-level-img/out", this->gpu, this->cuda_env);
    BS::thread_pool pool = BS::thread_pool(CUDA_STREAMS);

    private:

    void passtrough(Actor& actor, ConvNetwork& conv, float* preds, float* cpu_final, float* prob_end, uint32_t i, cudaStream_t stream);

    //performs Stochastic Policy Action Selection
    uint32_t pick_action(float* cpu_mem);

    //calculates all the advantages, which is basically just
    //values (from critic) sub returns =>
    //                At = Vt - Rt
    void calculate_advantage(Critic critic, float* advn);

    //calculates all Rt (r=> reward, returns Rt = rt + discount(gamma)*Rt+1)
    void calculate_return(float* returns);

    void learn();

    void reset_activations();

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
