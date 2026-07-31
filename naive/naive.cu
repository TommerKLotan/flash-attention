#include <torch/extension.h>
#include <cuda_runtime.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define IND(x, y, cols) ((y) * (cols) + (x))
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
    int sum = 0;
    for (int i = 0; i < seq_len; i ++)
    {
        sum += query_mat[IND(x, i, q_embed_dim)] * key_mat[IND(y, i, q_embed_dim)];
    }
    scores_mat[IND(x, y, seq_len)] = sum;

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

    dim3 threads_per_block(1024);
    dim3 number_of_blocks(batch_size, num_heads);

    calc_score_matrix<<<number_of_blocks, threads_per_block>>>(
        query.data_ptr<float>(), key.data_ptr<float>(), scores.data_ptr<float>(), seq_len, q_embed_dim);

    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("naive_attention", &naive_attention, "Naive attention (CUDA)");
}