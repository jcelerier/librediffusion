/** SANA-Streaming V2V DiT — in-tree module (GDN CUDA kernel + bf16 DiT forward).
 *
 * Ported from the validated standalone pipeline (lrd-bench-5090): the GDN
 * recurrence kernel + the full 20-block DiT forward (x_embedder -> blocks ->
 * final), fp32, matching the PyTorch reference to ~3.1% rel-L2 vs the bf16
 * golden. Weights are loaded from a directory of raw fp32 .bin files produced
 * by dit-block-test/export_model.py.
 *
 * Public interface is plain C++ (no CUDA types) so librediffusion_c.sana.cpp
 * can wrap it for the DLL C-API.
 */
#pragma once

#include <string>
#include <vector>

namespace librediffusion
{

class SanaDiT
{
public:
  /** Load all top-level + 20-block weights (fp32 .bin) resident from `weights_dir`. */
  explicit SanaDiT(const std::string& weights_dir);
  ~SanaDiT();

  SanaDiT(const SanaDiT&) = delete;
  SanaDiT& operator=(const SanaDiT&) = delete;

  /** Run the full model forward (x_embedder -> 20 blocks -> final -> unpatchify)
   *  on the resident golden input; fill `out` (N*128 fp32, token-major) and, if a
   *  gold_model_out.bin is present, report rel-L2 vs it in `rel_l2` (else -1).
   *  Returns 0 on success. Used for the in-tree no-regression self-test. */
  int full_forward(std::vector<float>& out, float& rel_l2);

  /** One DiT denoise forward: x_embed (N*C fp32 device ptr, post x_embedder+pos)
   *  -> noise_out (N*128 fp32 device ptr). The 98.95 ms bf16-capable engine. */
  int dit_forward(const float* x_embed_dev, float* noise_out_dev);

  /** Full 5-chunk streaming rollout (cache carry + FlowMatchEuler scheduler) over the
   *  per-chunk data in `data_dir` (roll_c0..roll_c4 from export_stream5.py). Reports the
   *  full 16-frame latent rel-L2 vs the Python sampler's gen in `rel_l2`. Returns 0 on ok. */
  int run_rollout(const std::string& data_dir, float& rel_l2);
  int run_rollout_sc(
      const std::string& data_dir, float& rel_l2); // self-contained cache production

  /** Set the TRT engine dir (vae/*.plan, gemma/*.plan) and the v2v data dir
   *  (latents_mean/std.bin, init_noise.bin) used by run_v2v. */
  int set_trt(const std::string& trt_dir, const std::string& data_dir);
  /** Tokenized prompt (int64 ids+mask, len 300) -> gemma -> stored text embeds. */
  int encode_prompt_ids(const long long* ids, const long long* mask);
  /** Directly set the (300,2304) gemma text embeds (host) — bypasses gemma, for
   *  reproducing a reference prompt encoding exactly in tests. */
  int set_text_embeds(const float* embeds);
  /** Full V2V: in_frames RGB8 (121*480*832*3) -> out_frames RGB8 (same); sets *n_out=121.
   *  Pipeline: preprocess -> VAE-encode -> self-contained 5-chunk rollout -> VAE-decode. */
  int run_v2v(
      const unsigned char* in_frames, int n_in, unsigned char* out_frames, int* n_out);

  int N() const; // spatial-temporal tokens (1560 for the reference chunk)
  int C() const; // hidden size (2240)

private:
  struct Impl;
  Impl* p_;
};

} // namespace librediffusion
