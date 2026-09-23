__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val,
                    __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__global__ void flashattention(const float* Q, const float* K, const float* V,
    float* O,
    int N,
    int d)
{
    // 教学版：
    // 一个 block 负责一个 Query row
    // 一个 warp 处理一个 K/V tile
    // 因此这里建议 blockDim.x = 32

    int row  = blockIdx.x;
    int lane = threadIdx.x;

    const int BK = 32;

    if (row >= N)
        return;

    // ============================================================
    // Online Softmax 状态
    // ============================================================

    float m = -INFINITY;   // 已处理 score 的最大值
    float l = 0.0f;        // 已处理 score 的 softmax 分母

    /*
        为了让代码容易理解，这里每个线程负责
        输出向量 O[row] 的若干维。

        例如：
        d = 1024
        32 threads

        lane 0:
            dim = 0,32,64,...

        lane 1:
            dim = 1,33,65,...

        ...
    */

    // 这里为了教学简单，假设 d <= 1024
    // 每个线程最多保存 32 个输出维度
    const int MAX_PER_THREAD = 32;

    float acc[MAX_PER_THREAD];

    int count = 0;

    for (int dim = lane; dim < d && count < MAX_PER_THREAD; dim += 32)
    {
        acc[count++] = 0.0f;
    }


    // ============================================================
    // 遍历所有 K/V tile
    // ============================================================

    for (int tile_start = 0;
         tile_start < N;
         tile_start += BK)
    {
        // --------------------------------------------------------
        // 1. 当前线程负责哪个 Key？
        // --------------------------------------------------------

        int j = tile_start + lane;

        float score = -INFINITY;

        if (j < N)
        {
            score = 0.0f;

            // Q[row] · K[j]

            for (int k = 0; k < d; ++k)
            {
                score +=
                    Q[row * d + k] *
                    K[j   * d + k];
            }

            score /= sqrtf((float)d);
        }


        // --------------------------------------------------------
        // 2. 当前 tile 最大值
        // --------------------------------------------------------

        float tile_max = warpReduceMax(score);

        // Reduce 后只有 lane 0 有最终结果
        tile_max =
            __shfl_sync(
                0xffffffff,
                tile_max,
                0
            );


        // --------------------------------------------------------
        // 3. 更新全局 running max
        // --------------------------------------------------------

        float m_new = fmaxf(m, tile_max);


        /*
            这是 FlashAttention / Online Softmax
            最重要的一行之一。

            旧数据原来使用：

                exp(score - m)

            现在最大值变成 m_new。

            所以旧数据必须重新缩放：

                exp(score - m_new)

              = exp(score - m)
                *
                exp(m - m_new)
        */

        float alpha = expf(m - m_new);


        // --------------------------------------------------------
        // 4. 当前 score 的指数
        // --------------------------------------------------------

        float p = 0.0f;

        if (j < N)
        {
            p = expf(score - m_new);
        }


        // --------------------------------------------------------
        // 5. 当前 tile softmax 分母
        // --------------------------------------------------------

        float tile_sum = warpReduceSum(p);

        tile_sum =
            __shfl_sync(
                0xffffffff,
                tile_sum,
                0
            );


        // --------------------------------------------------------
        // 6. 更新 Online Softmax 分母
        // --------------------------------------------------------

        float l_new = l * alpha + tile_sum;
        // --------------------------------------------------------
        // 7. 更新 O 的 numerator
        // --------------------------------------------------------

        /*
            我们维护的其实不是最终 O：

                    Σ exp(score - m) V
                O = --------------------
                    Σ exp(score - m)

            acc 保存的是上面的 numerator。

            最大值从 m -> m_new 后：

                old_acc *= exp(m - m_new)

            然后加入当前 tile：

                + Σ p_j V_j
        */

        int idx = 0;

        for (int dim = lane;
             dim < d && idx < count;
             dim += 32)
        {
            // 先修正旧 accumulator
            float new_acc = acc[idx] * alpha;

            /*
                当前 tile 中：

                Σ_j p_j * V[j][dim]

                注意：

                当前线程负责输出 dim，
                因此需要遍历 tile 中所有 j。
            */

            for (int t = 0; t < BK; ++t)
            {
                int key_index = tile_start + t;

                if (key_index < N)
                {
                    /*
                        当前写法为了教学清晰，
                        重新计算 score。

                        工业实现当然不会这么干，
                        score / probability 会通过
                        shared memory / register tiling
                        复用。
                    */

                    float s = 0.0f;

                    for (int k = 0; k < d; ++k)
                    {
                        s +=
                            Q[row * d + k] *
                            K[key_index * d + k];
                    }

                    s /= sqrtf((float)d);

                    float prob =
                        expf(s - m_new);

                    new_acc +=
                        prob *
                        V[key_index * d + dim];
                }
            }

            acc[idx] = new_acc;

            ++idx;
        }


        // --------------------------------------------------------
        // 8. 更新状态
        // --------------------------------------------------------

        m = m_new;
        l = l_new;
    }


    // ============================================================
    // 最终归一化
    // ============================================================

    int idx = 0;

    for (int dim = lane;
         dim < d && idx < count;
         dim += 32)
    {
        O[row * d + dim] =
            acc[idx] / l;

        ++idx;
    }
}
