#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_CPU_UTIL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_CPU_UTIL_H_

#include <vector>

#include "absl/random/random.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {
void SetGlobalTemperature(float t);

// Computes the top k indices of the given logits. The logits must be a 2D
// tensor (in a flattened buffer) of shape [batch_size, vocab_size]. The output
// is a vector of indices of shape [batch_size, k].
absl::StatusOr<std::vector<int>> TopKIndicies(absl::Span<const float> logits,
                                              int k, int batch_size = 1);

// Computes the softmax of the given logits.
//   - logits: a 2D tensor (in a flattened buffer) of shape
//     [batch_size, vocab_size].
//   - topk_indices: a 2D tensor (in a flattened buffer) of shape
//     [batch_size, k]. The indices of the top k logits.
//   - temperature: the temperature of the softmax.
//   - batch_size: the batch size of the logits.
//   - max_logit_values: this is an output parameter to store the max logit
//     values of each batch. It is a vector of shape [batch_size].
// The output is a vector of probabilities of shape [batch_size, vocab_size].
absl::StatusOr<std::vector<float>> Softmax(
    absl::Span<const float> logits, absl::Span<const int> topk_indices,
    float temperature, int batch_size, std::vector<float>& max_logit_values);

// Samples a batch of token ids from the given probabilities.
//   - logits: a 2D tensor (in a flattened buffer) of shape
//     [batch_size, vocab_size].
//   - k: the number of top k.
//   - p: the probability threshold use by Top-P sampling.
//   - temperature: the temperature used for calculating the softmax.
//   - rng: the random generator.
//   - batch_size: the batch size of the logits.
//   - sampled_scores: this is an output parameter to store the sampled scores
//     (as probabilities between 0 and 1) of each batch. It is a vector of shape
//     [batch_size]. Note that the probabilities is only an approximation of the
//     true probabilities as they are calculated based on the top-k logits
//     which are not normalized across the entire vocab. When k == 1, the
//     sampled_scores are always 1.0.
absl::StatusOr<std::vector<int>> TopKTopPSampling(
    absl::Span<const float> logits, int k, float p, float temperature,
    absl::BitGen& rng, int batch_size, std::vector<float>& sampled_scores);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_SAMPLING_CPU_UTIL_H_
