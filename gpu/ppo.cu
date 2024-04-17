#ifndef WARP_SIZE
    #define WARP_SIZE 32
#endif

__global__
void ppo_v1(float* out, float* prob, float* prob_old, float* adv, float* idx_cur, float* idx_prev, float epsilon, int n_elms) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int nth_elm = y * WARP_SIZE + x;

    if (nth_elm < n_elms) {

        float pred_cur = prob[(int)idx_cur[nth_elm]];
        float pred_prev = prob_old[(int)idx_prev[nth_elm]];

        float rt = pred_cur / pred_prev;
        float advantage = adv[nth_elm];

        float min_left = rt * advantage;

        float clip = rt;
        
        if (clip < (1 - epsilon)) {
            clip = 1 - epsilon;
        }
        else if (clip > (1 + epsilon)) {
            clip = 1 + epsilon;
        }
        
        float min_right = clip * advantage;
        float min_final = min(min_left, min_right);

        out[nth_elm] = min_final;
    }

}



