#pragma once
#include "./allocator/allocator.hpp"
#include "layers/convolution/conv_layer.hpp"
#include "layers/dense/dense_layer.hpp"
#include "layers/max_pooling/max_pooling.hpp"
#include "../data/environment/environment.hpp"
#include "../thread-pool/pool.hpp"
#include <random>

#define BATCH_SIZE 32

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

#include <cstdint>

class Actor {

private:
    //Possible outputs => 3
    //do softmax on CPU
    DenseLayer l1_64_64;
    DenseLayer l2_64_3;

    size_t gradient_size;
    size_t input_size;
    size_t output_size;

public:

    void act(uint32_t* cnn_processed, float* out);

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

    void value(float* cnn_processed, float* out);

    Critic();
    ~Critic();
    Critic(const Critic& other);
    Critic(const Critic&& other);

    //first we calculate the loss
    //lt = (Vt - Rt)^2
    //sum them up and use the scalar to calulate the gradient with
    //respect to the loss
    void calculate_gradient(float* values, float* returns, float* out);

    //prob will do SGM maybe Adam idk we'll see but this 
    //shouldn't be an issue
    void apply_gradient(float* gradients, float loss);

};

//include conv_layer_biases, max pooling, first without and then with
class ConvNetwork {

private:

    //soon to come layer definitions and stuff, will be implemented
    //not now tho a bit too lazy rn
    //not lazy anymore 0:
    ConvolutionalLayer l1_13x13_16x3x3; //ReLU activation
    ConvolutionalLayer l2_11x11_32x4x4; //ReLU activation
    MaxPooling l3_8x8_2x2;
    ConvolutionalLayer l4_4x4_32x3x3; //ReLU activation
    DenseLayer l5_2x2x32_64;

    size_t gradient_size;
    size_t input_size;
    size_t output_size;

public:

    void pass(float* state, float* out);

    ConvNetwork();
    ~ConvNetwork();
    ConvNetwork(const ConvNetwork& other);
    ConvNetwork(const ConvNetwork&& other);

    //this will be calculated two times each training steps 
    //due to Actor and Critic shared architecture
    //once done with Actors final gradients, then with Critics
    void calculate_gradient(float* input_gradients, float* out);

    //prob will do SGM maybe Adam idk we'll see but this 
    //shouldn't be an issue
    void apply_gradient(float* gradients, float loss);
};

class RLAgent {

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

    float* cuda_layer_weights;
    float* cuda_layer_biases;
    float* cuda_current_batch;
    float* cuda_current_env = nullptr;

    //theta/theta_old
    //should be of size network activations * batch_size
    float* cuda_activations;
    float* old_cuda_activations;

    //copied over from cuda_activations, this is only for confort
    //3 extra floats wont kill anybody
    //this will be on CPU
    float* cuda_final_predict_pass;

    float* cuda_values;
    float* cuda_returns;

    float* cuda_actor_gradient;
    float* cuda_critic_gradient;
    float* cuda_cnn_gradient;

    uint32_t weights_total;
    uint32_t biases_total;
    uint32_t batch_size;
    size_t current_batch;
    size_t current_frame;
    size_t current_reward;

    size_t actor_gradient_size;
    size_t critic_gradient_size;
    size_t cnn_gradient_size;

    //For Stochastic Policy Action Selection
    std::random_device rd;
    std::mt19937 gen; 
    std::uniform_real_distribution<float> dist;


    GPU::Device gpu = GPU::Device(0);
    //we align for 128(4 * 32) bits since thats the most GPU warps can mem access at a time
    Allocator alloc = Allocator(this->gpu, sizeof(float)*4); 
    Environment env = Environment("../data/parse-mario-level-img/out", this->gpu, this->cuda_current_env);
    ThreadPool::Pool pool = ThreadPool::Pool(16);

    private:

    void passtrough(Actor actor, ConvNetwork conv, float* preds);

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
