#include "./network.hpp"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include <iostream>
#include <new>

void RLAgent::reset_activations() {

    auto res = this->gpu.memset(this->cuda_activations, 0.0, activations_total * BATCH_SIZE);
    this->gpu.memset(this->cuda_activation_z, 0.0, activations_total * BATCH_SIZE);
    this->gpu.memset(this->cuda_actor_gradient, 0.0, actor_gradient_size * BATCH_SIZE);
    this->gpu.memset(this->cuda_critic_gradient, 0.0, critic_gradient_size * BATCH_SIZE);
    this->gpu.memset(this->cuda_cnn_gradient, 0.0, cnn_gradient_size * BATCH_SIZE);
    this->gpu.memset(this->cuda_gradients_with_respect_out, 0.0, grad_with_out * BATCH_SIZE);

    if (res != cudaSuccess) {

        std::cerr << "RLAgent::reset_activations | Couldnt set memory :(" << std::endl;
        exit(-1);
    }

}

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
        //SET EVERYTHING IN ACTV AND STUFF TO ZERO :))))
        //Dont forget to reset indexes in max pooling too otherwise big bad :(
        //for now Im too lazy and my head hurts :(
        this->reset_activations();

        //DONT FORGET TO RESET ENV AFTER THIS, CURRENT REWARD ALSO
        this->current_reward = 0;
        this->env.rindex = 0;
        this->current_frame = 1;
        //exit(-1);
    }

    std::cout << "Current frame: " << this->current_frame - 1 << " BATCH: " << this->batch_size << std::endl;

    //move to other thread before blocking main with our prediction :)
    //utilize the power of a different stream !

    pool.detach_task(
           [this] {

            //std::cout << "Address: " << this->old_cuda_activations + (current_frame-1) * activations_total << std::endl;
            passtrough(
                this->copy_actor, 
                this->copy_conv, 
                this->old_cuda_activations + (current_frame-1) * activations_total, 
                this->copy_final_predict_pass, 
                cuda_all_prob_old,
                current_frame-1,
                this->streams[1]
            ); 

            float agent_choice = pick_action(copy_final_predict_pass);

            auto res = gpu.memcpy_host(&agent_choice, old_cuda_action_taken + current_frame-1, sizeof(float));

            if (res != cudaSuccess) {
            std::cout << "RLAgent::predict | Error while copying current policy choice over" << std::endl;
            }
        }
    );
    //do a passtrough on a the current state and move data to CPU
    //of course, using the newest actor and conv network
    //-1 because this thing is 1 into the future sort of

    passtrough(
            this->actor, 
            this->conv, 
            this->cuda_activations + (current_frame-1) * activations_total, 
            this->cpu_final_predict_pass,
            cuda_all_prob_cur,
            current_frame-1,
            this->streams[0]
            );

    //COMMENT MOVE activations stuff etc

    pool.detach_task(
            [this] {
            //calc discounts and values here D:
            critic.value(
                    this->cuda_activations + (current_frame-1) * (cnn_activations_footprint+actor_activations_footprint) + cnn_activations_footprint,
                    this->cuda_activations + BATCH_SIZE * (cnn_activations_footprint+actor_activations_footprint) + (current_frame-1) * critic_activations_footprint, 
                    this->streams[2]
                    ); 

            //move Values to the locations this->cuda_values

            auto res = cudaMemcpy(
                    cuda_values + (current_frame-1),
                    this->cuda_activations + BATCH_SIZE*activations_total + current_frame*critic_activations_footprint - 1, 
                    sizeof(float), 
                    cudaMemcpyDeviceToDevice
                    );

            if (res != cudaSuccess) {
            std::cerr << "RLAgent::predict | Copy value from loc 1 to loc 2 failed! " << std::endl;
            exit(-1);
            }


            //cant do discounts D:
            }
    );

    //this basically just performs softmax over newest predictions we recorded
    //memsets the softmax to gpu and returns picked option, the one with highest
    //probability calculated by softmax
    float agent_choice = this->pick_action(this->cpu_final_predict_pass);
    auto res = gpu.memcpy_host(&agent_choice, cuda_action_taken + current_frame-1, sizeof(float));

    if (res != cudaSuccess) {
        std::cout << "RLAgent::predict | Error while copying current policy choice over" << std::endl;
    }

    return agent_choice;
}

uint32_t RLAgent::pick_action(float* cpu_mem) {

    //first perform softmax

    float max_logit = *std::max_element(this->cpu_final_predict_pass, this->cpu_final_predict_pass + AGENT_NUM_ACTIONS);

    float sum_of_exp = 0.0f;
    for (size_t i = 0; i < AGENT_NUM_ACTIONS; ++i) {
        //std::cout << "Final Activations[" << i << "]: " << cpu_final_predict_pass[i] << std::endl;
        cpu_mem[i] = std::exp(cpu_mem[i] - max_logit);
        sum_of_exp += cpu_mem[i];
    }

    for (size_t i = 0; i < AGENT_NUM_ACTIONS; ++i) {
        cpu_mem[i] /= sum_of_exp;
    }

    for (size_t i = 0; i < AGENT_NUM_ACTIONS; ++i) {
        //std::cout << "Prob[" << i << "] = " << cpu_final_predict_pass[i] << std::endl;
    }

    float rng_number = this->dist(this->gen);
    float cumulative_probability = 0.0f;

    for (size_t i = 0; i < this->actor_out; ++i) {

        cumulative_probability += cpu_mem[i];

        if (cumulative_probability > rng_number) {
            return i;
        }
    }

    return actor_out - 1;
}

void RLAgent::passtrough(Actor& actor, ConvNetwork& conv, float* preds, float* cpu_final, float* prob_end, uint32_t i, cudaStream_t stream) {

    //nah this will only work as intended, the other stuff will get coded in later 
    //like its simple I have my own reference to work with
    float* cuda_frame = this->cuda_env + i*MAP_SIZE;

    GPU::Tensor input = GPU::Tensor { cuda_frame, CNN_L1_IN, CNN_L1_IN, CNN_L1_IN_DEPTH };
    GPU::Tensor output = GPU::Tensor { preds, CNN_L1_OUT, CNN_L1_OUT, CNN_L2_OUT_DEPTH };

    conv.pass(input.dat_pointer, output.dat_pointer, stream);

    input = GPU::Tensor {preds + cnn_activations_footprint - CNN_L5_OUT,
        ACTOR_L1_IN, ACTOR_L1_IN, ACTOR_L1_IN_DEPTH};

    output = GPU::Tensor {preds + cnn_activations_footprint, 
        ACTOR_L1_OUT, ACTOR_L1_OUT, ACTOR_L1_OUT_DEPTH};

    actor.act(input.dat_pointer, output.dat_pointer, stream);

    //gpu.device_sync();
    cudaStreamSynchronize(stream);

    //copy prediction over to CPU mem so we can later pick action

    float* src = preds + cnn_activations_footprint + 
        actor_activations_footprint - 
        ACTOR_L2_OUT;

    //COMMENT BIG NEWS, DIVIDE BY 4 TO GET THE ACTUAL ELEMENT DIFFERENCE
    //BECAUSE IT PRINTS BYTES, NOT ELEMS OR WHATEVER :)

    //Possible error I may be overwritting this when calling predict old 
    //not very likely but still a possibility
    int res = this->gpu.memcpy_device(src, cpu_final,
            sizeof(float) * AGENT_NUM_ACTIONS);

    if (res != cudaSuccess) {

        std::cerr << "RLAgent::passtrough | Error copying data from gpu over to cpu!" << std::endl;
    }

    res = cudaMemcpy(prob_end, src, sizeof(float) * AGENT_NUM_ACTIONS, cudaMemcpyDeviceToDevice);

    if (res != cudaSuccess) {

        std::cerr << "RLAgent::passtrough | Error copying data from gpu over to gpu!" << std::endl;
    }

    cudaStreamSynchronize(stream);
    //function to perform softmax !!!

}

void RLAgent::learn() {

    //wait for all the threads to finish before moving on with the stuff C:
    pool.wait();

    for (auto& stream: streams) {
        cudaStreamSynchronize(stream);
    }

    this->copy_conv.deep_copy(gpu, conv);
    this->copy_actor.deep_copy(gpu, actor);

    //NOTE IT CRASHED FOR SOME REASON IDK WHY PROBABLY SOME INVALID ACCESS THAT EVENTUALL
    //HAPPENED SINCE NO ERROR MESSAGE!
    //okay only seems to be happening in the memory sanetizer

    //later used for critic loss
    auto res = gpu.memcpy_device(cuda_rewards, cpu_rewards, sizeof(float)*BATCH_SIZE);

    if (res != cudaSuccess) {
        std::cerr << "RLAgent::learn | Error while copying memory from device to host (rewards)!" << std::endl;
    }

    float running_sum = cpu_rewards[BATCH_SIZE - 1];

    //omg havent written any of these in a while while
    for (int i = BATCH_SIZE - 2; i >= 0; --i) {
        running_sum = cpu_rewards[i] + RLAGENT_GAMMA * running_sum;
        cpu_rewards[i] = running_sum;
    }

    res = gpu.memcpy_host(cpu_rewards, cuda_discounted_rewards, sizeof(float)*BATCH_SIZE);

    if (res != cudaSuccess) {
        std::cerr << "RLAgent::learn | Error while copying memory from cpu to device (rewards)!" << std::endl;
    }

    //DO PPO PART OF THINGS
    GPU::Tensor gae_delta_out = GPU::Tensor {
        this->cuda_gae_delta,
        BATCH_SIZE,
        1,
        1,
    };

    GPU::Tensor rewards = GPU::Tensor {
        this->cuda_rewards,
        BATCH_SIZE,
        1,
        1,
    };

    GPU::Tensor values = GPU::Tensor {
        this->cuda_values,
        BATCH_SIZE,
        1,
        1,
    };

    gpu.gae_delta(gae_delta_out, rewards, values, RLAGENT_GAMMA, this->streams[0]);
    cudaStreamSynchronize(streams[0]);

    GPU::Tensor gae_full_out = GPU::Tensor {
        this->cuda_gae,
        BATCH_SIZE,
        1,
        1,
    };

    gpu.gae_full(gae_full_out, gae_delta_out, RLAGENT_GAMMA, RLAGENT_LAMBDA, this->streams[0]);
    cudaStreamSynchronize(streams[0]);

    GPU::Tensor out = GPU::Tensor {
        this->cuda_ppo_objective,
        BATCH_SIZE, 
        1, 
        1
    };

    GPU::Tensor prob_cur = GPU::Tensor {
        this->cuda_all_prob_cur,
        BATCH_SIZE*AGENT_NUM_ACTIONS, 
        1, 
        1
    };

    GPU::Tensor prob_prev = GPU::Tensor {
        this->cuda_all_prob_old,
        BATCH_SIZE*AGENT_NUM_ACTIONS, 
        1, 
        1
    };


    GPU::Tensor idx_cur = GPU::Tensor {
        this->cuda_action_taken,
        BATCH_SIZE, 
        1, 
        1
    };

    GPU::Tensor idx_prev = GPU::Tensor {
        this->old_cuda_action_taken,
        BATCH_SIZE, 
        1, 
        1
    };

    gpu.ppo(out, prob_cur, prob_prev, gae_full_out, idx_cur, idx_prev, RLAGENT_EPSILON, this->streams[0]);
    cudaStreamSynchronize(streams[0]);

    GPU::Tensor mse_out = GPU::Tensor {
        this->cuda_critic_mse,
        BATCH_SIZE, 
        1, 
        1
    };

    GPU::Tensor discounted_rewards = GPU::Tensor {
        this->cuda_discounted_rewards,
        BATCH_SIZE, 
        1, 
        1
    };

    gpu.mse_der(mse_out, discounted_rewards, values, streams[0]);
    cudaStreamSynchronize(streams[0]);

        
    int items_per_thread = BATCH_SIZE / THREAD_POOL_SIZE;

    for (uint32_t id = 0; id < THREAD_POOL_SIZE; ++id) {
        for (uint32_t i = id*items_per_thread; i < (id+1)*items_per_thread; ++i) {

            pool.detach_task([this, mse_out, i] {

                cudaStream_t stream = streams[i%CUDA_STREAMS];

                    //TIME FOR GRADIENT :)
                    //critic first and then actor + cNN
                GPU::Tensor mse = GPU::Tensor {
                    mse_out.dat_pointer + i,
                        1,
                        1,
                        1
                };

                GPU::Tensor g_loss_wrt_c_out = GPU::Tensor {
                    cuda_gradients_with_respect_out + 
                    BATCH_SIZE*(cnn_activations_footprint+actor_activations_footprint) +
                    i*actor_gradual_l2,
                    1, 
                    1, 
                    1,
                };

                GPU::Tensor critic_output_z = GPU::Tensor {
                    cuda_activation_z + 
                    BATCH_SIZE*(cnn_activations_footprint+actor_activations_footprint) +
                    i * critic_activations_footprint - CRITIC_L2_OUT,
                    1,
                    1,
                    1,
                };

                GPU::Tensor td = GPU::Tensor {
                    cuda_gae_delta + i,
                    1,
                    1,
                    1
                };

                gpu.matsub(g_loss_wrt_c_out, critic_output_z, td, stream);

                //Critic output layer first
                GPU::Tensor grad_wrt_critic_output_z = GPU::Tensor {
                    g_loss_wrt_c_out.dat_pointer,
                    1,
                    1,
                    1,
                };

                GPU::Tensor grad_wrt_critic_out_w = GPU::Tensor {
                    cuda_critic_gradient + i*critic_gradient_size,
                    CRITIC_L2_OUT,
                    CRITIC_L1_OUT,
                    1
                };

                GPU::Tensor critic_output_layer_input = GPU::Tensor {
                    this->cuda_activations + 
                    i*(cnn_activations_footprint+actor_activations_footprint) + 
                    cnn_activations_footprint,
                    CNN_L1_OUT,
                    CNN_L2_OUT,
                    1
                };

                gpu.vector_outer(
                    grad_wrt_critic_out_w,
                    critic_output_layer_input,
                    grad_wrt_critic_output_z,
                    stream
                );

                GPU::Tensor critic_grad_wrt_out_bias = GPU::Tensor {
                    cuda_critic_gradient + BATCH_SIZE * critic_gradient_size + i*(CRITIC_L1_OUT + CRITIC_L2_OUT),
                    1,
                    1,
                    1,
                };

                cudaMemcpy(
                    critic_grad_wrt_out_bias.dat_pointer, 
                    grad_wrt_critic_output_z.dat_pointer,
                    sizeof(float),
                    cudaMemcpyDeviceToDevice
                );

                //Critic second layer (first layer but you get it we're going backwards :))
                
                GPU::Tensor critic_dense_actv_z = GPU::Tensor {
                    critic_output_z.dat_pointer - CRITIC_L1_OUT,
                    CRITIC_L1_OUT,
                    1,
                    1
                };

                GPU::Tensor critic_l1_inp = GPU::Tensor {
                    cuda_activations + i*(cnn_activations_footprint+actor_activations_footprint)+ 
                    cnn_activations_footprint - CNN_L5_OUT,
                    CRITIC_L1_IN,
                    1,
                    1
                };

                GPU::Tensor critic_l1_w = GPU::Tensor {
                    critic.l1_64_64.cudaMat, 
                    CRITIC_L1_IN, 
                    CRITIC_L1_OUT,
                    1
                };

                GPU::Tensor critic_l2_w = GPU::Tensor {
                    critic.l2_64_1.cudaMat, 
                    CRITIC_L2_IN, 
                    CRITIC_L2_OUT,
                    1
                };

                GPU::Tensor critic_l1_b = GPU::Tensor {
                    critic.l1_64_64.cudaBias, 
                    CRITIC_L1_IN, 
                    CRITIC_L1_OUT,
                    1
                };

                gpu.preactivations_dense_relu(
                    critic_dense_actv_z,
                    critic_l1_inp,                    
                    critic_l1_w,
                    critic_l1_b,
                    stream
                );

                GPU::Tensor grad_loss_wrt_critic_dense_out = GPU::Tensor { 
                    cuda_gradients_with_respect_out + 
                    BATCH_SIZE*(cnn_activations_footprint+actor_activations_footprint) +
                    i*critic_gradual_l2+critic_grad_l1,
                    CRITIC_L1_OUT, 
                    1, 
                    1,
                };

                gpu.matmul_ver1_gpu(
                    grad_wrt_critic_output_z.dat_pointer,
                    critic_l2_w.dat_pointer,
                    grad_loss_wrt_critic_dense_out.dat_pointer,
                    grad_wrt_critic_output_z.dat_x,
                    grad_wrt_critic_output_z.dat_y,
                    critic_l2_w.dat_x,
                    critic_l2_w.dat_y,
                    grad_loss_wrt_critic_dense_out.dat_x,
                    grad_loss_wrt_critic_dense_out.dat_y, 
                    GPU::ActivationFunction::None,
                    stream
                );

                GPU::Tensor grad_loss_wrt_critic_dense_z = GPU::Tensor {
                    cuda_gradients_with_respect_out + 
                    BATCH_SIZE*(cnn_activations_footprint+actor_activations_footprint) +
                    i*critic_gradual_l2+critic_grad_l1,
                    CRITIC_L1_OUT, 
                    1, 
                    1,
                };

                gpu.matmul_elementwise(
                    grad_loss_wrt_critic_dense_out,
                    critic_dense_actv_z,
                    grad_loss_wrt_critic_dense_z,
                    stream,
                    GPU::ActivationFunction::None
                );

                GPU::Tensor grad_wrt_critic_dense_w_delta = GPU::Tensor {
                    cuda_critic_gradient + i*critic_gradient_size + critic_grad_l1,
                    CRITIC_L1_OUT,
                    CNN_L5_OUT,
                    1
                };

                gpu.vector_outer(
                    grad_wrt_critic_dense_w_delta, 
                    critic_l1_inp, 
                    grad_loss_wrt_critic_dense_z, 
                    stream
                );

                GPU::Tensor critic_grad_wrt_dense_b_delta = GPU::Tensor {
                    cuda_critic_gradient + BATCH_SIZE * critic_gradient_size + i*(CRITIC_L1_OUT + CRITIC_L2_OUT) + CRITIC_L2_OUT,
                    CRITIC_L1_OUT,
                    1,
                    1,
                };

                cudaMemcpy(
                    critic_grad_wrt_dense_b_delta.dat_pointer, 
                    grad_loss_wrt_critic_dense_z.dat_pointer,
                    sizeof(float),
                    cudaMemcpyDeviceToDevice
                );

                GPU::Tensor cnn_grad_wrt_out_critic = GPU::Tensor {
                    cuda_gradients_with_respect_out + 
                    i*(cnn_activations_footprint+actor_activations_footprint) +
                    actor_gradual_l2,
                    CRITIC_L1_IN, 
                    1, 
                    1,
                };

                /*
                std::cout << "glwrt x: " << grad_loss_wrt_critic_dense_z.dat_x 
                    << " y: " << grad_loss_wrt_critic_dense_z.dat_y << std::endl; 

                std::cout << "critic_w x: " << critic_l1_w.dat_x 
                    << " y: " << critic_l1_w.dat_y << std::endl; 
                */

                gpu.matmul_ver1_gpu(
                    grad_loss_wrt_critic_dense_z.dat_pointer,
                    critic_l1_w.dat_pointer,
                    cnn_grad_wrt_out_critic.dat_pointer,
                    grad_loss_wrt_critic_dense_z.dat_x,
                    grad_loss_wrt_critic_dense_z.dat_y,
                    critic_l1_w.dat_x,
                    critic_l1_w.dat_y,
                    grad_loss_wrt_critic_dense_out.dat_x,
                    grad_loss_wrt_critic_dense_out.dat_y, 
                    GPU::ActivationFunction::None,
                    stream
                );

                //ACTOR PART (GRADIENTS YAAAY)
                //dl/dz
                GPU::Tensor actor_dl_dz = GPU::Tensor {
                    cuda_gradients_with_respect_out + 
                    i*(cnn_activations_footprint+actor_activations_footprint) +
                    actor_gradual_l2,
                    AGENT_NUM_ACTIONS,
                    1, 
                    1 
                };

                cudaMemcpy(
                    actor_dl_dz.dat_pointer,
                    cuda_all_prob_cur + i*AGENT_NUM_ACTIONS,
                    sizeof(float) * AGENT_NUM_ACTIONS,
                    cudaMemcpyDeviceToDevice
                );

                float agent_action_taken = 0;

                gpu.memcpy_device(cuda_action_taken + i, &agent_action_taken, sizeof(float));

                GPU::Tensor actor_dl_dz_sub_one = GPU::Tensor {
                    actor_dl_dz.dat_pointer + (uint32_t)agent_action_taken,
                    1,
                    1,
                    1,
                };

                gpu.subs_number(actor_dl_dz_sub_one, +1, stream);  

                float ppo_scalar = 0.0f;

                auto res = gpu.memcpy_device(cuda_ppo_objective + i, &ppo_scalar, sizeof(float));

                if (res != cudaSuccess) {
                    std::cerr << "RLAgent::learn | couldn't copy advantage scalar" << std::endl;
                }

                gpu.vector_scalar(
                    actor_dl_dz, 
                    actor_dl_dz, 
                    -ppo_scalar, 
                    stream
                );

                GPU::Tensor actor_grad_loss_actor_output_w = GPU::Tensor {
                    cuda_actor_gradient + i*actor_gradient_size,
                    ACTOR_L2_IN,
                    AGENT_NUM_ACTIONS,
                    1
                };

                GPU::Tensor actor_dense_activations = GPU::Tensor {
                    cuda_activations + i*(actor_activations_footprint + cnn_activations_footprint) +
                    cnn_activations_footprint - ACTOR_L1_IN,
                    ACTOR_L1_OUT,
                    1,
                    1
                };

                gpu.vector_outer(actor_grad_loss_actor_output_w, actor_dense_activations, actor_dl_dz, stream);

                GPU::Tensor actor_grad_loss_actor_output_b = GPU::Tensor {
                    cuda_actor_gradient + BATCH_SIZE*actor_gradient_size + i*actor_gradual_l2,
                    AGENT_NUM_ACTIONS,
                    1,
                    1
                };

                res = cudaMemcpy(
                    actor_grad_loss_actor_output_b.dat_pointer,
                    actor_dl_dz.dat_pointer, 
                    sizeof(float) * AGENT_NUM_ACTIONS, 
                    cudaMemcpyDeviceToDevice
                );

                if (res != cudaSuccess) {
                    std::cerr << "RLAgent::learn | Error while copying actor output bias gradient" << std::endl;
                }

                cudaMemcpy(
                    cuda_gradients_with_respect_out + (i+1)*(actor_activations_footprint + cnn_activations_footprint) -
                    AGENT_NUM_ACTIONS,
                    actor_dl_dz.dat_pointer,
                    sizeof(float) * AGENT_NUM_ACTIONS,
                    cudaMemcpyDeviceToDevice
                );

                if (res != cudaSuccess) {
                    std::cerr << "RLAgent::learn | Error while copying actor output gradient wrt its output" << std::endl;
                }

                //ACTOR SECOND LAYER :)
                GPU::Tensor grad_loss_wrt_actor_dense_actv = GPU::Tensor {
                    cuda_gradients_with_respect_out + (i+1)*(actor_activations_footprint + cnn_activations_footprint) -
                    AGENT_NUM_ACTIONS - ACTOR_L1_OUT,
                    ACTOR_L1_OUT,
                    1,
                    1
                };

                GPU::Tensor actor_output_w = GPU::Tensor { 
                    actor.l2_64_4.cudaMat,
                    ACTOR_L2_IN,
                    ACTOR_L2_OUT,
                    1,
                };

                gpu.matmul_ver1_gpu(
                    actor_dl_dz.dat_pointer,
                    actor_output_w.dat_pointer,
                    grad_loss_wrt_actor_dense_actv.dat_pointer,
                    actor_dl_dz.dat_x,
                    actor_dl_dz.dat_y,
                    actor_output_w.dat_x,
                    actor_output_w.dat_y,
                    grad_loss_wrt_actor_dense_actv.dat_x,
                    grad_loss_wrt_actor_dense_actv.dat_y, 
                    GPU::ActivationFunction::None,
                    stream
                );

                GPU::Tensor actor_dense_input = GPU::Tensor {
                    cuda_activations + i*(actor_activations_footprint + cnn_activations_footprint) +
                    cnn_activations_footprint - ACTOR_L1_IN,
                    ACTOR_L1_OUT,
                    1,
                    1
                };

                GPU::Tensor actor_dense_preactivations = GPU::Tensor {
                    cuda_activation_z + i*(actor_activations_footprint + cnn_activations_footprint) +
                    cnn_activations_footprint,
                    ACTOR_L1_OUT,
                    1,
                    1
                };

                GPU::Tensor actor_l1_w = GPU::Tensor {
                    actor.l1_64_64.cudaMat,
                    ACTOR_L1_IN,
                    ACTOR_L1_OUT,
                    1,
                };

                GPU::Tensor actor_l1_b = GPU::Tensor {
                    actor.l1_64_64.cudaBias,
                    ACTOR_L1_OUT,
                    1,
                    1,
                };

                //compute the fing preactivations yaaay I love this so so much!
                gpu.preactivations_dense_relu(
                    actor_dense_preactivations, 
                    actor_dense_input, 
                    actor_l1_w, 
                    actor_l1_b, 
                    stream 
                );

                gpu.matmul_elementwise(
                    grad_loss_wrt_actor_dense_actv, 
                    actor_dense_input, 
                    grad_loss_wrt_actor_dense_actv, 
                    stream, 
                    GPU::ActivationFunction::None
                );

                GPU::Tensor actor_dense_delta_w = GPU::Tensor {
                    cuda_actor_gradient + i*actor_gradient_size + AGENT_NUM_ACTIONS*ACTOR_L2_IN,
                    ACTOR_L1_IN,
                    ACTOR_L1_OUT,
                    1

                };

                gpu.vector_outer(
                    actor_dense_delta_w, 
                    actor_dense_input, 
                    grad_loss_wrt_actor_dense_actv, 
                    stream
                );

                GPU::Tensor actor_dense_delta_b = GPU::Tensor {
                    cuda_actor_gradient + BATCH_SIZE*actor_gradient_size + i*actor_gradual_l2 + AGENT_NUM_ACTIONS,
                    ACTOR_L1_OUT,
                    1,
                    1
                };

                res = cudaMemcpy(
                    actor_dense_delta_b.dat_pointer,
                    grad_loss_wrt_actor_dense_actv.dat_pointer, 
                    sizeof(float) * ACTOR_L1_OUT, 
                    cudaMemcpyDeviceToDevice
                );


                GPU::Tensor cnn_grad_wrt_out_actor = GPU::Tensor {
                    cuda_actor_grad_wrt_in + i*ACTOR_L1_IN,
                    CRITIC_L1_IN, 
                    1, 
                    1,
                };

                gpu.matmul_ver1_gpu(
                    grad_loss_wrt_actor_dense_actv.dat_pointer,
                    actor_l1_w.dat_pointer,
                    cnn_grad_wrt_out_actor.dat_pointer,
                    grad_loss_wrt_actor_dense_actv.dat_x,
                    grad_loss_wrt_actor_dense_actv.dat_y,
                    actor_l1_w.dat_x,
                    actor_l1_w.dat_y,
                    cnn_grad_wrt_out_actor.dat_x,
                    cnn_grad_wrt_out_actor.dat_y, 
                    GPU::ActivationFunction::None,
                    stream
                );

                //OKAY CNN FINALLY
                //aggregate stuff yk
                gpu.matadd_ver1(
                    cnn_grad_wrt_out_actor.dat_pointer,
                    cnn_grad_wrt_out_critic.dat_pointer,
                    cnn_grad_wrt_out_critic.dat_pointer,
                    cnn_grad_wrt_out_actor.dat_x,
                    cnn_grad_wrt_out_actor.dat_y, 
                    cnn_grad_wrt_out_critic.dat_x,
                    cnn_grad_wrt_out_critic.dat_y,
                    cnn_grad_wrt_out_critic.dat_x,
                    cnn_grad_wrt_out_critic.dat_y, 
                    stream
                );

                //cnn dense layer
                GPU::Tensor cnn_dense_preactivations = GPU::Tensor {
                    cuda_activation_z + i*(actor_activations_footprint + cnn_activations_footprint) +
                    cnn_activations_footprint - CNN_L5_OUT,
                    CNN_L5_OUT,
                    1,
                    1
                };

                GPU::Tensor cnn_l5_w = GPU::Tensor {
                    conv.l5_2x2x32_64.cudaMat,
                    CNN_L5_OUT,
                    CNN_L5_IN*CNN_L5_IN*CNN_L5_IN_DEPTH, 
                    1,
                };

                GPU::Tensor cnn_l5_b = GPU::Tensor {
                    conv.l5_2x2x32_64.cudaBias,
                    CNN_L5_OUT,
                    1,
                    1
                };

                GPU::Tensor cnn_l5_input = GPU::Tensor {
                    cuda_activations + i*(actor_activations_footprint + cnn_activations_footprint) +
                    cnn_activations_footprint - CNN_L5_OUT - CNN_L4_IN*CNN_L4_IN*CNN_L4_IN_DEPTH,
                    CNN_L5_IN*CNN_L5_IN*CNN_L5_IN_DEPTH, 
                    1,
                    1
                };

                gpu.preactivations_dense_relu(
                    cnn_dense_preactivations,
                    cnn_l5_input,
                    cnn_l5_w,
                    cnn_l5_b,
                    stream
                );

                gpu.matmul_elementwise(
                    cnn_dense_preactivations, 
                    cnn_grad_wrt_out_critic,
                    cnn_grad_wrt_out_critic,
                    stream,
                    GPU::ActivationFunction::None
                );

                GPU::Tensor cnn_dense_l5_grad_wrt_w = GPU::Tensor {
                    cuda_cnn_gradient + i*cnn_gradient_size, //64*128 => number of weights of the last layer
                    64*128,
                    1,
                    1
                };

                GPU::Tensor cnn_dense_l5_grad_wrt_b = GPU::Tensor {
                    cuda_cnn_gradient + BATCH_SIZE*cnn_gradient_size + 
                    i*cnn_gradual_l5,
                    CNN_L5_OUT,
                    1,
                    1
                };

                //Ok so issue is that Im totally dumb and just most likely didnt notice the thing and
                //also it seemed a bit goofy to me that it was like that
                gpu.vector_outer(
                    cnn_dense_l5_grad_wrt_w,
                    cnn_l5_input,
                    cnn_grad_wrt_out_critic,
                    stream
                );

                res = cudaMemcpy(
                    cnn_dense_l5_grad_wrt_b.dat_pointer,
                    cnn_grad_wrt_out_critic.dat_pointer, 
                    sizeof(float) * CNN_L5_OUT, 
                    cudaMemcpyDeviceToDevice
                );

                //what am I doing man this doesnt really matter, I dont need it
                //idk why Im doing this, just dont everwrite something :)
                GPU::Tensor cnn_l5_grad_wrt_inp = GPU::Tensor {
                    cuda_gradients_with_respect_out + i*(actor_activations_footprint + cnn_activations_footprint),
                    CNN_L5_IN*CNN_L5_IN*CNN_L5_IN_DEPTH,
                    1,
                    1
                };

                gpu.matmul_ver1_gpu(
                    cnn_grad_wrt_out_critic.dat_pointer,
                    cnn_l5_w.dat_pointer,
                    cnn_l5_grad_wrt_inp.dat_pointer,
                    cnn_grad_wrt_out_critic.dat_x,
                    cnn_grad_wrt_out_critic.dat_y, 
                    cnn_l5_w.dat_y,
                    cnn_l5_w.dat_x,
                    cnn_l5_grad_wrt_inp.dat_x,
                    cnn_l5_grad_wrt_inp.dat_y, 
                    GPU::None,
                    stream
                );


                //ok now compute cnn pre activations ! yaay
                //also this does not matter what specific index this is
                //only things that do matter are it not interfering with other or gradients wrt w/b
                GPU::Tensor cnn_l4_preactivations = GPU::Tensor {
                    cuda_activation_z + i*(actor_activations_footprint + cnn_activations_footprint),
                    CNN_L4_OUT,
                    CNN_L4_OUT,
                    CNN_L4_OUT_DEPTH,
                };

                GPU::Tensor cnn_l4_input = GPU::Tensor {
                    cuda_activations + i*(actor_activations_footprint + cnn_activations_footprint) +
                    cnn_activations_footprint - CNN_L5_OUT - 128 - 
                    CNN_L4_IN*CNN_L4_IN*CNN_L4_IN_DEPTH,
                    CNN_L4_IN,
                    CNN_L4_IN,
                    CNN_L4_IN_DEPTH
                };

                GPU::Tensor cnn_l4_w = GPU::Tensor {
                    conv.l4_4x4_32x3x3.cuda_kernel,
                    3,
                    3,
                    32
                };

                GPU::Tensor cnn_l4_b = GPU::Tensor {
                    conv.l4_4x4_32x3x3.cuda_bias,
                    32,
                    1,
                    1
                };

                gpu.conv_ver2_preactivations(
                    cnn_l4_preactivations,
                    cnn_l4_input, 
                    cnn_l4_w,
                    cnn_l4_b,
                    stream
                );

                GPU::Tensor cnn_l4_loss_wrt_inp = GPU::Tensor {
                    cuda_gradients_with_respect_out + i*(actor_activations_footprint + cnn_activations_footprint),
                    128,
                    1,
                    1
                };

                GPU::Tensor cnn_l4_gradient_wrt_b = GPU::Tensor {
                    cuda_cnn_gradient + BATCH_SIZE*cnn_gradient_size + i*cnn_gradual_l5 + CNN_L5_OUT_DEPTH,
                    32,
                    1,
                    1

                };

                gpu.sum_bias_cnn(cnn_l4_gradient_wrt_b, cnn_l5_grad_wrt_inp, 3, stream);

                gpu.matmul_elementwise(
                    cnn_l4_preactivations, 
                    cnn_l5_grad_wrt_inp, 
                    cnn_l4_loss_wrt_inp, 
                    stream, 
                    GPU::None
                );

                cnn_l4_loss_wrt_inp.dat_x = CNN_L4_OUT;
                cnn_l4_loss_wrt_inp.dat_y = CNN_L4_OUT;
                cnn_l4_loss_wrt_inp.dat_z = CNN_L4_OUT_DEPTH;

                
                GPU::Tensor cnn_l4_gradient_wrt_w = GPU::Tensor {
                    cuda_cnn_gradient + i*cnn_gradient_size + 64*128, //64*128 => number of weights of the last layer
                    3,
                    3,
                    32
                };

                gpu.conv_ver2_all(
                    cnn_l4_gradient_wrt_w, 
                    cnn_l4_input, 
                    cnn_l4_loss_wrt_inp, 
                    stream
                );

                GPU::Tensor cnn_l4_g_wrt_inp_f = GPU::Tensor {
                    cuda_gradients_with_respect_out + i*(actor_activations_footprint + cnn_activations_footprint) + 130,
                    CNN_L4_IN,
                    CNN_L4_IN,
                    CNN_L4_IN_DEPTH,
                };

                gpu.full_convolution(
                    cnn_l4_g_wrt_inp_f, 
                    cnn_l4_loss_wrt_inp, 
                    cnn_l4_w, 
                    stream
                );

                //max pooling
                GPU::Tensor cnn_l3_g_wrt_inp = GPU::Tensor {
                    cuda_gradients_with_respect_out + i*(actor_activations_footprint + cnn_activations_footprint) + 130 + CNN_L4_IN*CNN_L4_IN*CNN_L4_IN_DEPTH,
                    CNN_L3_IN,
                    CNN_L3_IN,
                    CNN_L3_IN_DEPTH
                };

                int* indices = conv.l3_8x8_2x2.cuda_idx + i*CNN_L3_OUT*CNN_L3_OUT*CNN_L3_OUT_DEPTH;

                gpu.max_pool_der(cnn_l3_g_wrt_inp, cnn_l4_g_wrt_inp_f, indices, stream); 
                
                //cnn 2
                
                GPU::Tensor cnn_l2_preactivations = GPU::Tensor {
                    cuda_activation_z + i*(actor_activations_footprint + cnn_activations_footprint),
                    CNN_L2_OUT,
                    CNN_L2_OUT,
                    CNN_L2_OUT_DEPTH,
                };

                GPU::Tensor cnn_l2_input = GPU::Tensor {
                    cuda_activations + i*(actor_activations_footprint + cnn_activations_footprint) +
                    cnn_activations_footprint - CNN_L5_OUT - 128 - 
                    CNN_L4_IN*CNN_L4_IN*CNN_L4_IN_DEPTH - CNN_L3_IN*CNN_L3_IN*CNN_L3_IN_DEPTH -
                    CNN_L2_IN*CNN_L2_IN*CNN_L2_IN_DEPTH,
                    CNN_L2_IN,
                    CNN_L2_IN,
                    CNN_L2_IN_DEPTH
                };

                GPU::Tensor cnn_l2_w = GPU::Tensor {
                    conv.l2_11x11_32x4x4.cuda_kernel,
                    4,
                    4,
                    32
                };

                GPU::Tensor cnn_l2_b = GPU::Tensor {
                    conv.l2_11x11_32x4x4.cuda_bias,
                    32,
                    1,
                    1
                };

                gpu.conv_ver2_preactivations(
                    cnn_l2_preactivations,
                    cnn_l2_input, 
                    cnn_l2_w,
                    cnn_l2_b,
                    stream
                );

                GPU::Tensor cnn_l2_loss_wrt_inp = GPU::Tensor {
                    cuda_gradients_with_respect_out + i*(actor_activations_footprint + cnn_activations_footprint),
                    CNN_L2_IN*CNN_L2_IN*CNN_L2_IN_DEPTH,
                    1,
                    1
                };

                GPU::Tensor cnn_l2_gradient_wrt_b = GPU::Tensor {
                    cuda_cnn_gradient + BATCH_SIZE*cnn_gradient_size + i*cnn_gradual_l5 + CNN_L5_OUT_DEPTH + CNN_L3_OUT_DEPTH,
                    32,
                    1,
                    1

                };

                gpu.sum_bias_cnn(cnn_l2_gradient_wrt_b, cnn_l3_g_wrt_inp, 4, stream);

                gpu.matmul_elementwise(
                    cnn_l2_preactivations, 
                    cnn_l3_g_wrt_inp, 
                    cnn_l2_loss_wrt_inp, 
                    stream, 
                    GPU::None
                );

                cnn_l2_loss_wrt_inp.dat_x = CNN_L2_OUT;
                cnn_l2_loss_wrt_inp.dat_y = CNN_L2_OUT;
                cnn_l2_loss_wrt_inp.dat_z = CNN_L2_OUT_DEPTH;
                
                GPU::Tensor cnn_l2_gradient_wrt_w = GPU::Tensor {
                    cuda_cnn_gradient + i*cnn_gradient_size + 64*128 + 2*2*32, //64*128 => number of weights of the last layer
                    4,
                    4,
                    32
                };

                gpu.conv_ver2_all(
                    cnn_l2_gradient_wrt_w, 
                    cnn_l2_input, 
                    cnn_l2_loss_wrt_inp, 
                    stream
                );

                GPU::Tensor cnn_l2_g_wrt_inp_f = GPU::Tensor {
                    cuda_gradients_with_respect_out + i*(actor_activations_footprint + cnn_activations_footprint) + 130,
                    CNN_L2_IN,
                    CNN_L2_IN,
                    CNN_L2_IN_DEPTH,
                };

                gpu.full_convolution(
                    cnn_l2_g_wrt_inp_f, 
                    cnn_l2_loss_wrt_inp, 
                    cnn_l2_w, 
                    stream
                );
                
                //cnn 1

                 GPU::Tensor cnn_l1_preactivations = GPU::Tensor {
                    cuda_activation_z + i*(actor_activations_footprint + cnn_activations_footprint),
                    CNN_L1_OUT,
                    CNN_L1_OUT,
                    CNN_L1_OUT_DEPTH,
                };

                GPU::Tensor cnn_l1_input = GPU::Tensor {
                    cuda_activations + i*(actor_activations_footprint + cnn_activations_footprint) +
                    cnn_activations_footprint - CNN_L5_OUT - 128 - 
                    CNN_L4_IN*CNN_L4_IN*CNN_L4_IN_DEPTH - CNN_L3_IN*CNN_L3_IN*CNN_L3_IN_DEPTH -
                    CNN_L2_IN*CNN_L2_IN*CNN_L2_IN_DEPTH - CNN_L1_IN*CNN_L1_IN*CNN_L1_IN_DEPTH,
                    CNN_L1_IN,
                    CNN_L1_IN,
                    CNN_L1_IN_DEPTH
                };

                GPU::Tensor cnn_l1_w = GPU::Tensor {
                    conv.l1_13x13_16x3x3.cuda_kernel,
                    3,
                    3,
                    16
                };

                GPU::Tensor cnn_l1_b = GPU::Tensor {
                    conv.l1_13x13_16x3x3.cuda_bias,
                    16,
                    1,
                    1
                };

                gpu.conv_ver2_preactivations(
                    cnn_l1_preactivations,
                    cnn_l1_input, 
                    cnn_l1_w,
                    cnn_l1_b,
                    stream
                );

                GPU::Tensor cnn_l1_loss_wrt_inp = GPU::Tensor {
                    cuda_gradients_with_respect_out + i*(actor_activations_footprint + cnn_activations_footprint),
                    CNN_L1_IN*CNN_L1_IN*CNN_L1_IN_DEPTH,
                    1,
                    1
                };

                GPU::Tensor cnn_l1_gradient_wrt_b = GPU::Tensor {
                    cuda_cnn_gradient + BATCH_SIZE*cnn_gradient_size + i*cnn_gradual_l5 + 
                    CNN_L5_OUT_DEPTH + CNN_L3_OUT_DEPTH + CNN_L2_OUT_DEPTH,
                    16,
                    1,
                    1

                };

                gpu.sum_bias_cnn(cnn_l1_gradient_wrt_b, cnn_l2_g_wrt_inp_f, 3, stream);

                gpu.matmul_elementwise(
                    cnn_l1_preactivations, 
                    cnn_l2_g_wrt_inp_f, 
                    cnn_l1_loss_wrt_inp, 
                    stream, 
                    GPU::None
                );

                cnn_l1_loss_wrt_inp.dat_x = CNN_L1_OUT;
                cnn_l1_loss_wrt_inp.dat_y = CNN_L1_OUT;
                cnn_l1_loss_wrt_inp.dat_z = CNN_L1_OUT_DEPTH;
                
                GPU::Tensor cnn_l1_gradient_wrt_w = GPU::Tensor {
                    cuda_cnn_gradient + i*cnn_gradient_size + 64*128 + 2*2*32 + 4*4*32, //64*128 => number of weights of the last layer
                    3,
                    3,
                    16
                };

                gpu.conv_ver2_all(
                    cnn_l1_gradient_wrt_w, 
                    cnn_l1_input, 
                    cnn_l1_loss_wrt_inp, 
                    stream
                );

            });
        }
    }

    pool.wait();

    for (auto& s : streams) {
        cudaStreamSynchronize(s);
    }

    for (uint32_t id = 0; id < THREAD_POOL_SIZE; ++id) {
        for (uint32_t i = id*items_per_thread; i < (id+1)*items_per_thread; ++i) {

            pool.detach_task([this, mse_out, i] {

                cudaStream_t stream = streams[i%CUDA_STREAMS];

                gpu.matadd_ver1_simple( 
                    critic.l2_64_1.cudaMat,
                    cuda_critic_gradient + i*critic_gradient_size,
                    critic.l2_64_1.cudaMat,
                    64, 1,
                    stream);

                gpu.matadd_ver1_simple( 
                    critic.l1_64_64.cudaMat,
                    cuda_critic_gradient + i*critic_gradient_size +64,
                    critic.l1_64_64.cudaMat,
                    64*64, 1,
                    stream);

                gpu.matadd_ver1_simple( 
                    critic.l2_64_1.cudaBias,
                    cuda_critic_gradient + BATCH_SIZE*critic_gradient_size
                    + i*(64+1),
                    critic.l2_64_1.cudaBias,
                    1, 1,
                    stream);

                gpu.matadd_ver1_simple( 
                    critic.l1_64_64.cudaBias,
                    cuda_critic_gradient + BATCH_SIZE*critic_gradient_size
                    + i*(64+1)+1,
                    critic.l1_64_64.cudaBias,
                    64, 1,
                    stream);

                //ACTOR

                gpu.matadd_ver1_simple( 
                    actor.l2_64_4.cudaMat,
                    cuda_actor_gradient + i*actor_gradient_size,
                    actor.l2_64_4.cudaMat,
                    64*4, 1,
                    stream);

                gpu.matadd_ver1_simple( 
                    actor.l1_64_64.cudaMat,
                    cuda_actor_gradient + i*actor_gradient_size + 64*4,
                    actor.l1_64_64.cudaMat,
                    64*64, 1,
                    stream);

                gpu.matadd_ver1_simple( 
                    actor.l2_64_4.cudaBias,
                    cuda_actor_gradient + BATCH_SIZE*actor_gradient_size
                    + i*(64+4),
                    actor.l1_64_64.cudaBias,
                    4, 1,
                    stream);

                gpu.matadd_ver1_simple( 
                    actor.l1_64_64.cudaBias,
                    cuda_actor_gradient + BATCH_SIZE*actor_gradient_size
                    + i*(64+4)+4,
                    actor.l1_64_64.cudaBias,
                    64, 1,
                    stream);

                //CNN
                
                //L5
                gpu.matadd_ver1_simple( 
                    conv.l5_2x2x32_64.cudaMat,
                    cuda_cnn_gradient + i*cnn_gradient_size,
                    conv.l5_2x2x32_64.cudaMat,
                    128*64, 1,
                    stream);

                gpu.matadd_ver1_simple( 
                    conv.l5_2x2x32_64.cudaBias,
                    cuda_cnn_gradient + BATCH_SIZE*cnn_gradient_size
                    + i*(64+32+32+32+16),
                    conv.l5_2x2x32_64.cudaBias,
                    64, 1,
                    stream);

                //L4
                gpu.matadd_ver1_simple( 
                    conv.l4_4x4_32x3x3.cuda_kernel,
                    cuda_cnn_gradient + i*cnn_gradient_size+128*64,
                    conv.l4_4x4_32x3x3.cuda_kernel,
                    32*3*3, 1,
                    stream);

                gpu.matadd_ver1_simple( 
                    conv.l4_4x4_32x3x3.cuda_bias,
                    cuda_cnn_gradient + BATCH_SIZE*cnn_gradient_size
                    + i*(64+32+32+32+16) + 64,
                    conv.l4_4x4_32x3x3.cuda_bias,
                    32, 1,
                    stream);

                //L2
                gpu.matadd_ver1_simple( 
                    conv.l2_11x11_32x4x4.cuda_kernel,
                    cuda_cnn_gradient + i*cnn_gradient_size+128*64+32*3*3,
                    conv.l2_11x11_32x4x4.cuda_kernel,
                    32*4*4, 1,
                    stream);

                gpu.matadd_ver1_simple( 
                    conv.l2_11x11_32x4x4.cuda_bias,
                    cuda_cnn_gradient + BATCH_SIZE*cnn_gradient_size
                    + i*(64+32+32+32+16) + 64 + 32,
                    conv.l2_11x11_32x4x4.cuda_bias,
                    32, 1,
                    stream);

                //L1
                gpu.matadd_ver1_simple( 
                    conv.l1_13x13_16x3x3.cuda_kernel,
                    cuda_cnn_gradient + i*cnn_gradient_size+128*64+32*3*3+32*4*4,
                    conv.l1_13x13_16x3x3.cuda_kernel,
                    16*3*3, 1,
                    stream);

                gpu.matadd_ver1_simple( 
                    conv.l1_13x13_16x3x3.cuda_bias,
                    cuda_cnn_gradient + BATCH_SIZE*cnn_gradient_size
                    + i*(64+32+32+32+16) + 64 + 32 + 32,
                    conv.l1_13x13_16x3x3.cuda_bias,
                    16, 1,
                    stream);


            });
        }
    }

    pool.wait();

    for (auto& s : streams) {
        cudaStreamSynchronize(s);
    }

    std::cout << "Finished all training!" << std::endl;

    //calc softmax on ALL THE ACTIVATIONS 
    //including OLD/NEW network
    //DONT FORGET TO UPDATE OLD MODELS YK :) IDEALLY

    //also reset everything pls :)
    reset_activations();
}

RLAgent::RLAgent():
    actor_layers(2), actor_in(128), actor_out(AGENT_NUM_ACTIONS),
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
            GPU::mem_needed_align(sizeof(float) * 16 * 3 * 3 * 1, sizeof(float) * 4)/4 + 
            GPU::mem_needed_align(sizeof(float) * 32 * 4 * 4 * 16, sizeof(float) * 4)/4 + 
            //skip max pooling since nothing learnable there
            GPU::mem_needed_align(sizeof(float) * 32 * 3 * 3 * 32, sizeof(float) * 4)/4 + 
            GPU::mem_needed_align(sizeof(float) * 128 * 64, sizeof(float) * 4)/4;

        size_t critic_weights = 
            //critic network
            GPU::mem_needed_align(sizeof(float) * 64 * 64, sizeof(float) * 4)/4 + 
            GPU::mem_needed_align(sizeof(float) * 64 * 1, sizeof(float) * 4)/4;

        size_t actor_weights = GPU::mem_needed_align(sizeof(float) * 64 * 64, sizeof(float) * 4)/4 + 
            GPU::mem_needed_align(sizeof(float) * 64 * 3, sizeof(float) * 4)/4;

        this->weights_total = 
            //conv network first, and its layers
            cnn_weights + 
            //critic network
            critic_weights+ 
            //actor network
            actor_weights; 

        size_t cnn_biases = 
            //cnn
            (16 + 32 + 32 + 128); 

        size_t critic_biases = 
            //critic 
            64 + 1;

        size_t actor_biases = 
            //actor:wq
            64 + AGENT_NUM_ACTIONS;

        this->biases_total = GPU::mem_needed_align(
                sizeof(float) * (//cnn
                    cnn_biases + 
                    //critic
                    critic_biases + 
                    //actor
                    actor_biases), sizeof(float) * 4)/4;

        this->batch_size = 32;
        this->current_batch = 0;
        this->current_frame = 0;
        this->current_reward = 0;

        this->actor_gradient_size = GPU::mem_needed_align(
                sizeof(float) * (64*64 + 64*AGENT_NUM_ACTIONS + 64 + AGENT_NUM_ACTIONS), sizeof(float) * 4
                )/4;

        this->critic_gradient_size = GPU::mem_needed_align(
                sizeof(float) * (64*64 + 64*1 + 64 + 1), sizeof(float) * 4
                )/4;

        this->cnn_gradient_size = GPU::mem_needed_align(
                sizeof(float) * (144 + 16 + 8192 + 32 + 9216 + 32 + 8192 + 64), // Separating each component
                sizeof(float) * 4 // Alignment to 128 bits
                )/4;

        this->activations_total = GPU::mem_needed_align(
                sizeof(float) * (64 + AGENT_NUM_ACTIONS +
                    64 + 1 +
                    11 * 11 * 16 + 
                    8 * 8 * 32 +
                    4 * 4 * 32 +
                    2 * 2 * 32 +
                    64),
                sizeof(float) * 4 // Alignment to 128 bits
                )/4;

        this->grad_with_out = GPU::mem_needed_align(
                sizeof(float) * (AGENT_NUM_ACTIONS + 64 +
                    1 + 64 + 
                    64 + 2*2*32 + 4*4*32 + 8*8*32 + 11*11*16),
                //dense layer + conv layers
                sizeof(float) * 4 // Alignment to 128 bits
                )/4;

        size_t extra_mem = GPU::mem_needed_align(
                BATCH_SIZE                     + //PPO Objective
                BATCH_SIZE                     + //discounted rewards
                BATCH_SIZE                     + //gae_lambda
                BATCH_SIZE                     + //gae rewards
                1          + 1                 +  //sum_ppo, sum_critic_loss
                BATCH_SIZE * AGENT_NUM_ACTIONS + //OLD
                BATCH_SIZE * AGENT_NUM_ACTIONS + //NEW
                BATCH_SIZE                     +//MSE CRITIC 
                BATCH_SIZE * activations_total +//no activation func output
                BATCH_SIZE * ACTOR_L1_IN        //actor grad wrt in
                ,
                sizeof(float) * 4)/4;

        auto TOTAL_MEM_NEEDED = 
            this->weights_total     * 2              + //weight needed for network and network old
            this->biases_total      * 2              + //biases needed =||=
            this->activations_total * 2 * BATCH_SIZE + //cuda_actv_old + cuda_actv 
            MAP_SIZE                * BATCH_SIZE     + //env 
            BATCH_SIZE              * 4              + //advantage, values, returns, rewards
            grad_with_out           * BATCH_SIZE     + 
            cnn_gradient_size       * BATCH_SIZE     +
            critic_gradient_size    * BATCH_SIZE     +
            actor_gradient_size     * BATCH_SIZE     +
            extra_mem; 

        std::cout << "biases total: " << biases_total << std::endl;

        int res = alloc.alloc_new_block(GPU::mem_needed_align(
                    sizeof(float) * TOTAL_MEM_NEEDED
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
        float* big_block = alloc.alloc_space(sizeof(float) * TOTAL_MEM_NEEDED);

        std::cout << "Total MEM: " << TOTAL_MEM_NEEDED << std::endl;
        std::cout << "BLOCK RANGE " << big_block << " - " << big_block + TOTAL_MEM_NEEDED << std::endl;

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
        //std::cout << "Space for old: " << space_actv_old << std::endl;
        //std::cout << "Redundant print space for old: " << this->old_cuda_activations << std::endl;

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

        float* ppo_objective = big_block + p_offset;
        this->cuda_ppo_objective = ppo_objective;

        p_offset += BATCH_SIZE;

        float* discounted_rewards = big_block + p_offset;
        this->cuda_discounted_rewards = discounted_rewards;

        p_offset += BATCH_SIZE;

        float* gae_delta = big_block + p_offset;
        this->cuda_gae_delta = gae_delta;

        p_offset += BATCH_SIZE;

        float* gae = big_block + p_offset;
        this->cuda_gae = gae;

        p_offset += BATCH_SIZE;

        float* sum_ppo = big_block + p_offset;
        this->sum_ppo = sum_ppo;

        p_offset += 1;

        float* sum_critic_loss = big_block + p_offset;
        this->sum_critic_loss = sum_critic_loss;

        p_offset += 1;

        float* actions_old = big_block + p_offset;
        this->cuda_all_prob_old = actions_old;

        p_offset += BATCH_SIZE * AGENT_NUM_ACTIONS;

        float* actions_cur = big_block + p_offset;
        this->cuda_all_prob_cur = actions_cur;

        p_offset += BATCH_SIZE * AGENT_NUM_ACTIONS;

        float* mse_critic = big_block + p_offset;
        this->cuda_critic_mse = mse_critic;

        p_offset += BATCH_SIZE;

        float* activation_z = big_block + p_offset;
        this->cuda_activation_z = activation_z;

        p_offset += BATCH_SIZE * activations_total;

        float* actor_grad_wrt_in = big_block + p_offset;
        this->cuda_actor_grad_wrt_in = actor_grad_wrt_in;

        p_offset += BATCH_SIZE * ACTOR_L1_IN;

        //Ill just alloc separately idc 
        this->reset_activations();

        float* mem = gpu.allocate_memory(sizeof(float) * 2 * BATCH_SIZE);

        if (!mem) {
            std::cerr << "RLAgent::RLAgent | Error while allocating mem for agents choices!" << std::endl;
        }

        float* mem_for_old_choices = mem;
        this->old_cuda_action_taken = mem_for_old_choices;

        float* mem_for_current_choices = mem + BATCH_SIZE;
        this->cuda_action_taken = mem_for_current_choices;

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

        //COPY THE ORIGINAL IDEALLY :)
        //IMPLEMENT THE CORRECT COPY STUFF PLS
        this->copy_conv.init_self(this->gpu, weights_for_old + cuda_w_offset, biases_for_old + cuda_b_offset);
        cuda_w_offset += cnn_weights;
        cuda_b_offset += cnn_biases;

        this->copy_critic.init_self(this->gpu, weights_for_old + cuda_w_offset, biases_for_old + cuda_b_offset);
        cuda_w_offset += critic_weights;
        cuda_b_offset += critic_biases;

        this->copy_actor.init_self(this->gpu, weights_for_old + cuda_w_offset, biases_for_old + cuda_b_offset);
        cuda_w_offset += actor_weights;
        cuda_b_offset += actor_biases;

        copy_conv.deep_copy(this->gpu, this->conv);
        copy_actor.deep_copy(this->gpu, this->actor);

        //CPU MEM
        float* cpu_final_pp = new (std::nothrow) float[AGENT_NUM_ACTIONS];

        if (!cpu_final_pp) {
            std::cerr << "RLAgent::RLAgent() | Failed to allocate memory on the cpu!" << std::endl;
        }

        this->cpu_final_predict_pass = cpu_final_pp;

        float* old_cpu_final_pp = new (std::nothrow) float[AGENT_NUM_ACTIONS];

        if (!old_cpu_final_pp) {
            std::cerr << "RLAgent::RLAgent() | Failed to allocate memory on the cpu!" << std::endl;
        }

        this->copy_final_predict_pass = old_cpu_final_pp;

        float* cpu_rewards = new (std::nothrow) float[AGENT_NUM_ACTIONS];

        if (!cpu_rewards) {
            std::cerr << "RLAgent::RLAgent() | Failed to allocate memory on the cpu (rewards)!" << std::endl;
        }

        this->cpu_rewards = cpu_rewards;
        //Okay so everything may be initialized maybe? ?? please ?? ?? ?? ??? ?? ??
    }

////////////////////////////////////////////////////////
//////////////////////CNN CODE//////////////////////////
////////////////////////////////////////////////////////

ConvNetwork::ConvNetwork() {

}

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
    GPU::Tensor output = GPU::Tensor { cuda_out, CNN_L1_OUT, CNN_L1_OUT, CNN_L1_OUT_DEPTH };
    //go with nullptr just to verify something :)
    //the 'b' tensor is useless since its represented by the filters themself
    //if I want to do it this way I should make a static method that will do that 
    //might as well make use of the gpu API that I have 
    //its fine (not that bad as it used to be I improved it a bit)
    l1_13x13_16x3x3.convolve(input, GPU::Tensor {
            nullptr, 3, 3, 16 }, output.dat_pointer, stream);

    //shift out ptr and the in ptr
    //ugghh breaking safety stuff maybe but do I care? Hell nah
    //I can do whatever I want with my ptrs, whos gonna stop me
    input.dat_pointer = output.dat_pointer;
    input.dat_x = CNN_L1_OUT;
    input.dat_y = CNN_L1_OUT;
    input.dat_z = CNN_L1_OUT_DEPTH;
    output.dat_x = CNN_L2_OUT;
    output.dat_y = CNN_L2_OUT;
    output.dat_z = CNN_L2_OUT_DEPTH;
    output.dat_pointer = output.dat_pointer + CNN_L1_OUT*CNN_L1_OUT*CNN_L1_OUT_DEPTH;

    /*
       GPU::print_mem(input.dat_pointer, input.dat_x * input.dat_y * input.dat_z); 
       std::cout << "================================" << std::endl;
       GPU::print_mem(input.dat_pointer + input.dat_x * input.dat_y * input.dat_z, 10); 
       exit(-1);
       */

    //update input and output tensors with their respective #define thingy constant ?
    //okay they're all consts :|

    //Ok I got sidetracked but still Im happy with what progress I made today
    //I sort of found motivation again, maybe I should seek some help

    l2_11x11_32x4x4.convolve(input, GPU::Tensor {
            nullptr, 4, 4, 32 }, output.dat_pointer, stream);

    input.dat_x = CNN_L2_OUT;
    input.dat_y = CNN_L2_OUT;
    input.dat_z = CNN_L2_OUT_DEPTH;
    input.dat_pointer += CNN_L1_OUT*CNN_L1_OUT*CNN_L1_OUT_DEPTH;
    output.dat_x = input.dat_x / 2;
    output.dat_y = input.dat_y / 2;
    output.dat_z = CNN_L3_OUT_DEPTH;
    output.dat_pointer = output.dat_pointer + CNN_L2_OUT*CNN_L2_OUT*CNN_L2_OUT_DEPTH;

    /* 
       GPU::print_mem(input.dat_pointer, input.dat_x * input.dat_y * input.dat_z); 
       std::cout << "================================" << std::endl;
       GPU::print_mem(input.dat_pointer + input.dat_x * input.dat_y * input.dat_z, 10); 
       exit(-1);
       */

    l3_8x8_2x2.pool(input, output, stream);

    input.dat_x = CNN_L3_OUT;
    input.dat_y = CNN_L3_OUT;
    input.dat_z = CNN_L3_OUT_DEPTH;
    input.dat_pointer += CNN_L2_OUT*CNN_L2_OUT*CNN_L2_OUT_DEPTH;
    output.dat_x = CNN_L4_OUT;
    output.dat_y = CNN_L4_OUT;
    output.dat_z = CNN_L4_OUT_DEPTH;
    output.dat_pointer = output.dat_pointer + CNN_L3_OUT*CNN_L3_OUT*CNN_L3_OUT_DEPTH;

    /*
       GPU::print_mem(input.dat_pointer, input.dat_x * input.dat_y * input.dat_z); 
       std::cout << "================================" << std::endl;
       GPU::print_mem(input.dat_pointer + input.dat_x * input.dat_y * input.dat_z, 10); 
       exit(-1);
       */

    //ISSUE WAS HERE C:

    l4_4x4_32x3x3.convolve(input, GPU::Tensor {
            nullptr, 3, 3, 32 }, output.dat_pointer, stream);

    input.dat_x = CNN_L4_OUT;
    input.dat_y = CNN_L4_OUT;
    input.dat_z = CNN_L4_OUT_DEPTH;
    input.dat_pointer += CNN_L3_OUT*CNN_L3_OUT*CNN_L3_OUT_DEPTH;
    output.dat_x = CNN_L5_OUT;
    output.dat_y = 1;
    output.dat_z = CNN_L5_OUT_DEPTH;
    output.dat_pointer = output.dat_pointer + CNN_L4_OUT*CNN_L4_OUT*CNN_L4_OUT_DEPTH;

    /*
       GPU::print_mem(input.dat_pointer, input.dat_x * input.dat_y * input.dat_z); 
       std::cout << "================================" << std::endl;
       GPU::print_mem(input.dat_pointer + input.dat_x * input.dat_y * input.dat_z, 10); 
       exit(-1);
       */

    //I know a bug is somewhere over here I defo fucked up
    //Yeah there's at least one :(
    l5_2x2x32_64.passthrough(input.dat_pointer, output.dat_pointer, stream);

    //GPU::print_mem(output.dat_pointer, output.dat_x * 1 * output.dat_z); 
    //std::cout << "================================" << std::endl;
    //GPU::print_mem(input.dat_pointer + input.dat_x * input.dat_y * input.dat_z, 10); 
    //exit(-1);
    //std::cout << "Out ptr bef actor: " << output.dat_pointer << std::endl;
    //std::cout << "Weird test: " << out + cnn_activations_footprint << std::endl;
    //std::cout << "Test sum: " << test_sum << std::endl;
    cudaStreamSynchronize(stream);
}

void ConvNetwork::deep_copy(GPU::Device& gpu, const ConvNetwork& original) {
    l1_13x13_16x3x3.deep_copy(original.l1_13x13_16x3x3);
    l2_11x11_32x4x4.deep_copy(original.l2_11x11_32x4x4);
    l4_4x4_32x3x3.deep_copy(original.l4_4x4_32x3x3);
    l5_2x2x32_64.deep_copy(original.l5_2x2x32_64);
}

////////////////////////////////////////////////////////
////////////////////CRITIC CODE/////////////////////////
////////////////////////////////////////////////////////

Critic::Critic() {

}

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
            l2_neurons, l2_input, GPU::ActivationFunction::None, GPU::ActivationFunction::DerReLU);

}

void Critic::value(float* cnn_processed, float* out, cudaStream_t stream) {

    float* cuda_inp = cnn_processed;
    float* cuda_out = out;

    GPU::Tensor input = GPU::Tensor { cuda_inp, CRITIC_L1_IN, 1, CRITIC_L1_IN_DEPTH };
    GPU::Tensor output = GPU::Tensor { cuda_out, CRITIC_L1_OUT, 1, CRITIC_L1_OUT_DEPTH };

    l1_64_64.passthrough(input.dat_pointer, output.dat_pointer, stream);

    input.dat_x = CRITIC_L1_OUT;
    input.dat_y = 1;
    input.dat_z = CRITIC_L1_OUT_DEPTH;
    input.dat_pointer = output.dat_pointer;
    output.dat_x = CRITIC_L2_OUT;
    output.dat_y = 1;
    output.dat_z = CRITIC_L2_OUT_DEPTH;
    output.dat_pointer += CRITIC_L1_OUT*1*CRITIC_L1_OUT_DEPTH;

    l2_64_1.passthrough(input.dat_pointer, output.dat_pointer, stream);

    cudaStreamSynchronize(stream);
}

////////////////////////////////////////////////////////
////////////////////ACTOR CODE//////////////////////////
////////////////////////////////////////////////////////

Actor::Actor() {

}

void Actor::init_self(GPU::Device& gpu, float* cuda_w, float* cuda_b) {

    size_t weights_offset = 0;
    size_t biases_offset = 0;

    size_t l1_neurons = 64;
    size_t l1_input = 64;

    size_t l2_neurons = AGENT_NUM_ACTIONS;
    size_t l2_input = 64;

    l1_64_64.init_self(gpu, cuda_w + weights_offset, cuda_b + biases_offset, 
            l1_neurons, l1_input, GPU::ActivationFunction::ReLU, GPU::ActivationFunction::DerReLU);

    weights_offset += l1_input * l1_neurons;
    biases_offset += l1_neurons;

    l2_64_4.init_self(gpu, cuda_w + weights_offset, cuda_b + biases_offset, 
            l2_neurons, l2_input, GPU::ActivationFunction::ReLU, GPU::ActivationFunction::DerReLU);

}


void Actor::deep_copy(GPU::Device& gpu, const Actor& original) {
    this->l1_64_64.deep_copy(original.l1_64_64);
    this->l2_64_4.deep_copy(original.l2_64_4);
}

void Actor::act(float* cnn_processed, float* out, cudaStream_t stream) {

    float* cuda_inp = cnn_processed;
    float* cuda_out = out;

    GPU::Tensor input = GPU::Tensor { cuda_inp, ACTOR_L1_IN, 1, ACTOR_L1_IN_DEPTH };
    GPU::Tensor output = GPU::Tensor { cuda_out, ACTOR_L1_OUT, 1, ACTOR_L1_OUT_DEPTH };

    /*
       GPU::print_mem(input.dat_pointer, input.dat_x * 1 * input.dat_z); 
       std::cout << "================================" << std::endl;
       GPU::print_mem(input.dat_pointer + input.dat_x * 1 * input.dat_z, 10); 
       std::cout << "================================" << std::endl;
       exit(-1);
       */


    l1_64_64.passthrough(input.dat_pointer, output.dat_pointer, stream);

    //GPU::print_mem(output.dat_pointer, output.dat_x * 1 * output.dat_z); 
    //std::cout << "================================" << std::endl;


    input.dat_x = ACTOR_L1_OUT;
    input.dat_y = 1;
    input.dat_z = ACTOR_L1_OUT_DEPTH;
    input.dat_pointer = output.dat_pointer;
    output.dat_x = ACTOR_L2_OUT;
    output.dat_y = 1;
    output.dat_z = ACTOR_L2_OUT_DEPTH;
    output.dat_pointer = output.dat_pointer + ACTOR_L1_OUT*1*ACTOR_L1_OUT_DEPTH;

    /*
       GPU::print_mem(input.dat_pointer, input.dat_x * 1 * input.dat_z); 
       std::cout << "================================" << std::endl;
       GPU::print_mem(input.dat_pointer + input.dat_x * 1 * input.dat_z, 10); 
       std::cout << "================================" << std::endl;
       */

    l2_64_4.passthrough(input.dat_pointer, output.dat_pointer, stream);
    //std::cout << "OUTPUT(real out): " << out << std::endl;
    //std::cout << "OUTPUT: " << output.dat_pointer << std::endl;

    //GPU::print_mem(output.dat_pointer, output.dat_x * 1 * output.dat_z); 
    //exit(-1);

    cudaStreamSynchronize(stream);
}
