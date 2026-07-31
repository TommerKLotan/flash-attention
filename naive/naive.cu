#include <torch/extension.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define IND(r, c, cols) ((r) * (cols) + (c))
__global__ void calc_score_matrix(const float *query, const float *key, float *scores,
                                  int seq_len, int q_embed_dim)
{
    int batch_ind = blockIdx.y;
    int head_ind = blockIdx.z;
    int num_heads = gridDim.z;
    int start_offset = (batch_ind * num_heads + head_ind) * seq_len * q_embed_dim;
    const float *query_mat = query + start_offset;
    const float *key_mat = key + start_offset;
    float *scores_mat = scores + (batch_ind * num_heads + head_ind) * seq_len * seq_len;

    int item_offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (item_offset >= seq_len * seq_len)
    {
        return;
    }
    int x = item_offset % seq_len;
    int y = item_offset / seq_len;
    float sum = 0;
    for (int i = 0; i < q_embed_dim; i++)
    {
        sum += query_mat[IND(x, i, q_embed_dim)] * key_mat[IND(y, i, q_embed_dim)];
    }
    scores_mat[IND(x, y, seq_len)] = sum / sqrtf(q_embed_dim);
}

__global__ void softmax_scores(float *scores, int seq_len)
{
    int batch_ind = blockIdx.y;
    int head_ind = blockIdx.z;
    int num_heads = gridDim.z;
    int row = blockIdx.x;
    float *scores_mat = scores + (batch_ind * num_heads + head_ind) * seq_len * seq_len;
    __shared__ float sum, max;


    float tmp_max = -INFINITY;
    float val;
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < seq_len; i++)
        {
            val = scores_mat[IND(row, i, seq_len)];
            tmp_max = val > tmp_max ? val : tmp_max;
        }
    }
    max = tmp_max;
    __syncthreads();

    int chunk_size = CEIL_DIV(seq_len, blockDim.x);
    int range_start = chunk_size * threadIdx.x;
    int range_end = range_start + chunk_size;
    range_end = range_end <= seq_len ? range_end : seq_len;
    
    for (int i = range_start; i < range_end; i++)
    {
        scores_mat[IND(row, i, seq_len)] = expf(scores_mat[IND(row, i, seq_len)] - max);
    }

    __syncthreads();

    float tmp_sum = 0;
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < seq_len; i++)
        {
            tmp_sum += scores_mat[IND(row, i, seq_len)];
        }
    }
    sum = tmp_sum;
    __syncthreads();

    for (int i = range_start; i < range_end; i++)
    {
        scores_mat[IND(row, i, seq_len)] = scores_mat[IND(row, i, seq_len)] / sum;
    }

}

torch::Tensor naive_attention(torch::Tensor query, torch::Tensor key, torch::Tensor value)
{
    const auto batch_size = query.size(0);
    const auto num_heads = query.size(1);
    const auto seq_len = query.size(2);
    const auto q_embed_dim = query.size(3);
    const auto v_embed_dim = value.size(3);

    auto scores = torch::empty({batch_size, num_heads, seq_len, seq_len}, query.options());
    auto result = torch::empty_like(value);

    int num_blocks_per_score_mat = CEIL_DIV(seq_len * seq_len, 1024);
    dim3 score_threads_per_block(1024);
    dim3 score_number_of_blocks(num_blocks_per_score_mat, batch_size, num_heads);
    calc_score_matrix<<<score_number_of_blocks, score_threads_per_block>>>(
        query.data_ptr<float>(), key.data_ptr<float>(), scores.data_ptr<float>(), seq_len, q_embed_dim);

    dim3 softmax_threads_per_block(1024);
    dim3 softmax_number_of_blocks(seq_len, batch_size, num_heads);
    softmax_scores<<<softmax_number_of_blocks, softmax_threads_per_block>>>(scores.data_ptr<float>(), seq_len);

    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("naive_attention", &naive_attention, "Naive attention (CUDA)");
}