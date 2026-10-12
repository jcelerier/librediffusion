/**
 * FLUX.2-klein FPS harness — reproduces the ossia StreamDiffusion node's klein paths to measure FPS
 * WITHOUT needing the score/Qt build. It mirrors LibreDiffusion.cpp runKlein() (sync) and
 * runKleinAsync() (worker-thread diffusion on its own CUDA stream + render-thread RIFE on a separate
 * high-priority stream + per-tick phase interpolation), so the numbers track what users actually see.
 *
 * Modes (argv[9]):
 *   sync-noint   : node sync path, no interpolation  (diffuse + emit every tick)   -> diffusion fps
 *   sync-interp  : node sync path, RIFE decoupled queue (exp), STEADY-CLOCK ticked  -> displayed fps
 *   async        : node async PHASE-C path (dedicated producer thread + triple-buffer-style handoff;
 *                  render thread emits ONE precomputed sub-frame per steady tick)   -> displayed fps
 *                  + reports the underlying diffusion (keyframe) fps separately
 *
 * THE PACING BAR (why this harness exists): ossia ticks 30..1000fps; average fps HIDES stutter. What
 * matters is the per-frame INTERVAL distribution: a FLUID stream has tight intervals (low stddev,
 * p99 close to mean, small max-stall, few late frames). We measure mean/stddev/min/max/p50/p95/p99 +
 * late% + a histogram, and print a smoothness verdict.
 *
 * PHASE-C design under test (mirrors the node):
 *   - The diffusion keyframe (~150ms) runs on a background producer thread on the klein stream's OWN
 *     low-prio CUDA stream. It NEVER runs on the render/tick thread.
 *   - When a fresh keyframe pair (prev,cur) is adopted, the FULL RIFE sweep (2^exp sub-frames) is
 *     computed ONCE and cached. The per-tick render cost is then just a memcpy of the phase-indexed
 *     cached sub-frame -> constant, tiny, jitter-free.
 *   - The render tick runs on a STEADY wall-clock cadence; phase advances by real elapsed time scaled
 *     by the measured keyframe rate, so motion is paced to wall time and decoupled from when keyframes
 *     land. A keyframe is requested only when the sweep is ~consumed (leaves GPU headroom).
 *
 * Usage:
 *   klein_fps_harness <transformer.plan> <qwen.plan> <vae_decoder.plan> <vae_encoder.plan>
 *                     <tokenizer.json> <bn_mean.bin> <bn_std.bin> <rife.plan>
 *                     <mode> [exp=2] [render_fps=60] [W=320] [H=576]
 *  (for sync-noint, [render_fps] is reinterpreted as the tick COUNT.)
 *  env: LRD_KLEIN_STEPS (default 2), LRD_KLEIN_WARMUP (default 1), LRD_KLEIN_REF (raw WxH RGBA video),
 *       LRD_KLEIN_PROMPT, LRD_KLEIN_STRENGTH, LRD_KLEIN_MASK, LRD_CLIP_DIR, LRD_CLIP_MAX
 */
#include "librediffusion_c.h"
#include "qwen_tokenizer_c.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
static double now_s() {
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}
static std::vector<float> load_bin_f32(const char* p, size_t n) {
  std::ifstream f(p, std::ios::binary);
  std::vector<float> v(n);
  if(f) f.read((char*)v.data(), n * 4);
  return v;
}

// ----- pacing report (the real fluidity metric) -----------------------------------------------
static void report_pacing(const std::vector<double>& emit_ts)
{
  if(emit_ts.size() < 3) return;
  std::vector<double> iv;                       // per-frame intervals (ms)
  iv.reserve(emit_ts.size());
  for(size_t i = 1; i < emit_ts.size(); i++) iv.push_back((emit_ts[i] - emit_ts[i-1]) * 1000.0);
  std::sort(iv.begin(), iv.end());
  auto pct = [&](double p){ return iv[std::min(iv.size()-1, (size_t)(p * iv.size()))]; };
  double mean = 0; for(double v : iv) mean += v; mean /= iv.size();
  double var = 0; for(double v : iv) var += (v-mean)*(v-mean); var /= iv.size();
  double sd = std::sqrt(var);
  double mn = iv.front(), mx = iv.back();
  double med = pct(0.50), p95 = pct(0.95), p99 = pct(0.99);
  // deadline: a frame is "late" if its interval exceeds 1.5x the median (a visible hitch).
  double deadline = 1.5 * med; int late = 0; for(double v : iv) if(v > deadline) late++;
  printf("  --- frame pacing (per-frame interval, ms) ---\n");
  printf("    mean=%.2f  stddev=%.2f  (jitter)   min=%.2f  max=%.2f (worst stall)\n", mean, sd, mn, mx);
  printf("    p50=%.2f  p95=%.2f  p99=%.2f ms     => p99 fps=%.1f (worst-1%% sustained)\n",
         med, p95, p99, 1000.0/p99);
  printf("    late frames (>1.5x median): %d / %zu = %.1f%%  (lower=smoother)\n",
         late, iv.size(), 100.0 * late / iv.size());
  printf("    smoothness verdict: %s\n",
         (sd < 0.5*mean && mx < 3.0*med) ? "FLUID (tight pacing)"
         : (sd < mean) ? "ACCEPTABLE (some jitter)" : "STUTTERY (hitches dominate)");
  // histogram (interval ms buckets) — shows the bimodal "burst then stall" shape directly.
  static const double edges[] = {2,4,8,16,33,50,100,200,1e9};
  static const char* lbl[]   = {"<2","2-4","4-8","8-16","16-33","33-50","50-100","100-200",">200"};
  int hb[9] = {0};
  for(double v : iv){ int b=0; while(b<8 && v>edges[b]) b++; hb[b]++; }
  printf("    histogram (interval ms): ");
  for(int b=0;b<9;b++) if(hb[b]) printf("[%s]=%d ", lbl[b], hb[b]);
  printf("\n");
}

int main(int argc, char** argv)
{
  if(argc < 10)
  {
    fprintf(stderr,
            "usage: klein_fps_harness <transformer> <qwen> <vae_dec> <vae_enc> <tokenizer.json> "
            "<bn_mean.bin> <bn_std.bin> <rife.plan> <mode> [exp=2] [render_fps=60] [W=320] [H=576]\n"
            "  mode: sync-noint | sync-interp | async\n");
    return 1;
  }
  const char* transformer = argv[1];
  const char* qwen = argv[2];
  const char* vae_dec = argv[3];
  const char* vae_enc = argv[4];
  const char* tok = argv[5];
  const char* bnm = argv[6];
  const char* bns = argv[7];
  const char* rife_plan = argv[8];
  std::string mode = argv[9];
  const int exp = argc > 10 ? atoi(argv[10]) : 2;
  const int arg_fps = argc > 11 ? atoi(argv[11]) : 60;
  const int W = argc > 12 ? atoi(argv[12]) : 320;
  const int H = argc > 13 ? atoi(argv[13]) : 576;
  const double window_arg = argc > 14 ? atof(argv[14]) : 4.0;  // measurement window [s]
  // async present model: 0=smooth (FIFO), 1=fresh (newest-wins 1kf), 2=lowlatency (newest ASAP)
  const std::string pacing_arg = argc > 15 ? argv[15] : "smooth";
  const int pacing = (pacing_arg=="fresh") ? 1 : (pacing_arg=="lowlatency"||pacing_arg=="low") ? 2 : 0;
  const int Th = H / 16, Tw = W / 16;       // token grid (H=Th*16)
  const size_t nbytes = (size_t)W * H * 4;
  const char* PROMPT = std::getenv("LRD_KLEIN_PROMPT") ? std::getenv("LRD_KLEIN_PROMPT")
                                                       : "turn this into an oil painting, masterpiece";

  if(qwen_tokenizer_load(tok) != 0) { fprintf(stderr, "tokenizer load failed\n"); return 2; }

  auto* s = librediffusion_flux2_stream_create(transformer, qwen, vae_dec, vae_enc, tok, Th, Tw, 52ull, 0);
  if(!s) { fprintf(stderr, "stream_create failed\n"); return 3; }
  auto bn_mean = load_bin_f32(bnm, 128), bn_std = load_bin_f32(bns, 128);
  librediffusion_flux2_stream_set_bn(s, bn_mean.data(), bn_std.data());
  // LRD_KLEIN_STEPS: number of denoising steps (default 2, the node's default; 1 for few-step LoRAs)
  const int steps = std::getenv("LRD_KLEIN_STEPS") ? std::max(1, atoi(std::getenv("LRD_KLEIN_STEPS"))) : 2;
  librediffusion_flux2_stream_set_steps(s, steps);
  fprintf(stderr, "[klein] steps=%d\n", steps);
  if(librediffusion_flux2_stream_set_prompt(s, PROMPT) < 0) { fprintf(stderr, "set_prompt failed\n"); return 4; }

  // img2img strength (<1.0 = start from a noised real reference instead of pure noise)
  if(const char* st = std::getenv("LRD_KLEIN_STRENGTH")) {
    librediffusion_flux2_stream_set_strength(s, (float)atof(st));
    fprintf(stderr, "[klein] img2img strength=%s\n", st);
  }
  // inpaint mask (RGBA, white=regenerate / black=keep)
  if(const char* mp = std::getenv("LRD_KLEIN_MASK")) {
    std::ifstream mf(mp, std::ios::binary); std::vector<unsigned char> mask((size_t)W*H*4);
    if(mf && mf.read((char*)mask.data(), mask.size())) {
      librediffusion_flux2_stream_set_mask(s, mask.data(), H, W);
      fprintf(stderr, "[klein] inpaint mask loaded %s\n", mp);
    }
  }
  // real reference video (RGBA WxH frames) — replaces the synthetic pattern when LRD_KLEIN_REF is set
  std::vector<std::vector<unsigned char>> realref;
  if(const char* rp = std::getenv("LRD_KLEIN_REF")) {
    std::ifstream rf(rp, std::ios::binary);
    std::vector<unsigned char> fr(nbytes);
    while(rf.read((char*)fr.data(), nbytes)) realref.push_back(fr);
    fprintf(stderr, "[klein] real reference: %zu frames from %s\n", realref.size(), rp);
  }

  // RIFE (render-thread interpolation), high-priority stream owned inside the wrapper.
  librediffusion_rife_handle rife = nullptr;
  if(exp > 0)
  {
    rife = librediffusion_rife_create(rife_plan, 0);
    if(rife) { librediffusion_rife_set_enabled(rife, 1); librediffusion_rife_set_interpolation_exp(rife, exp); }
  }

  // a moving synthetic reference frame so each "real" frame genuinely differs (motion for RIFE).
  std::vector<unsigned char> ref(nbytes, 0), real_cur(nbytes, 0), real_prev(nbytes, 0), emit(nbytes, 0);
  auto fill_ref = [&](int t){
    if(!realref.empty()) { ref = realref[(size_t)t % realref.size()]; return; }
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){ long i=((long)y*W+x)*4;
      ref[i+0]=(unsigned char)((x + t*4) & 255); ref[i+1]=(unsigned char)((y) & 255);
      ref[i+2]=(unsigned char)((t*7) & 255); ref[i+3]=255; } };

  // set + first real frame
  // LRD_KLEIN_WARMUP: untimed frames before the window (default 1)
  const int warmup = std::getenv("LRD_KLEIN_WARMUP") ? std::max(1, atoi(std::getenv("LRD_KLEIN_WARMUP"))) : 1;
  for(int w = 0; w < warmup; w++) {
    fill_ref(0);
    librediffusion_flux2_stream_set_reference(s, ref.data());
    librediffusion_flux2_stream_frame_cached(s, real_cur.data());
  }

  int diffusions = 0, emitted = 0;
  std::vector<double> emit_ts;
  emit_ts.reserve((size_t)arg_fps * 8 + 4096);
  // optional clip capture: collect displayed frames into RAM during the timed window (no disk I/O / no
  // GPU work in the loop -> does not perturb fps), flushed to PPM after measurement. env LRD_CLIP_DIR.
  const char* CLIPDIR = std::getenv("LRD_CLIP_DIR");
  const size_t CLIP_MAX = std::getenv("LRD_CLIP_MAX") ? (size_t)atoi(std::getenv("LRD_CLIP_MAX")) : 2000;
  std::vector<std::vector<unsigned char>> clip;
  auto mark_emit = [&]{ emit_ts.push_back(now_s()); ++emitted;
                        if(CLIPDIR && clip.size() < CLIP_MAX) clip.emplace_back(emit); };
  double t_start = now_s();

  if(mode == "sync-noint")
  {
    // diffuse + emit every tick (full diffusion rate; interpolation off)
    const int ticks = arg_fps > 0 ? arg_fps : 60;
    for(int t = 0; t < ticks; t++) {
      fill_ref(t);
      librediffusion_flux2_stream_set_reference(s, ref.data());
      librediffusion_flux2_stream_frame_cached(s, emit.data());
      diffusions++; mark_emit();
    }
  }
  else if(mode == "sync-interp")
  {
    // node sync decoupled-queue, but ticked on a STEADY wall-clock (the real node is paced by score's
    // fixed-rate operator() — not a tight pop loop). Diffusion happens INLINE on the render thread when
    // the queue empties, so the stall is unavoidable on this path; this models the user-visible shape.
    const int target_render_fps = arg_fps > 0 ? arg_fps : 60;
    const double tick_dt = 1.0 / (double)target_render_fps;
    const double window_s = window_arg;
    std::vector<std::vector<unsigned char>> queue; size_t qpos = 0;
    bool have_prev = false;
    std::vector<unsigned char> prev_emit(nbytes,0); bool have_prev_emit=false;
    std::vector<double> motion_delta; motion_delta.reserve((size_t)arg_fps*16+4096);
    double w_start = now_s(), next_tick = w_start;
    while(now_s() - w_start < window_s) {
      if(qpos >= queue.size()) {
        real_prev = real_cur;
        fill_ref(diffusions+1);
        librediffusion_flux2_stream_set_reference(s, ref.data());
        librediffusion_flux2_stream_frame_cached(s, real_cur.data());     // BLOCKS the render loop (hitch)
        diffusions++;
        queue.clear(); qpos = 0;
        if(exp > 0 && rife && have_prev) {
          int nmax = 1 << exp; std::vector<unsigned char> scratch((size_t)nmax * nbytes); int n=0;
          if(librediffusion_rife_interpolate(rife, real_prev.data(), real_cur.data(), H, W, scratch.data(), &n)==LIBREDIFFUSION_SUCCESS && n>0)
            for(int i=0;i<n;i++) queue.emplace_back(scratch.begin()+(size_t)i*nbytes, scratch.begin()+(size_t)(i+1)*nbytes);
          else queue.push_back(real_cur);
        } else queue.push_back(real_cur);
        have_prev = true;
      }
      emit = queue[qpos++]; mark_emit();
      if(have_prev_emit){ double sm=0; for(size_t k=0;k<nbytes;k++) sm+=std::abs((int)emit[k]-(int)prev_emit[k]); motion_delta.push_back(sm/nbytes);}
      prev_emit=emit; have_prev_emit=true;
      next_tick += tick_dt;
      double sleep_s = next_tick - now_s();
      if(sleep_s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
    }
    t_start = now_s() - window_s;
    if(motion_delta.size()>3){ double m=0; for(double v:motion_delta)m+=v; m/=motion_delta.size();
      double var=0; for(double v:motion_delta)var+=(v-m)*(v-m); var/=motion_delta.size(); double sd=std::sqrt(var);
      int zeros=0; for(double v:motion_delta) if(v<0.05) zeros++;
      std::vector<double> srt=motion_delta; std::sort(srt.begin(),srt.end());
      printf("  --- MOTION smoothness (per-frame content delta) ---\n");
      printf("    mean=%.3f  stddev=%.3f  (rel=%.2f)  max=%.3f  dup-frames(<0.05)=%d/%zu=%.1f%%\n",
             m,sd,(m>1e-6?sd/m:0.0),srt.back(),zeros,motion_delta.size(),100.0*zeros/motion_delta.size()); }
  }
  else if(mode == "async")
  {
    // ===== PHASE C: dedicated producer thread + precomputed-sweep render loop =====
    // Producer thread diffuses on the klein stream's OWN (low-prio) CUDA stream. It is the ONLY caller
    // of the flux2 C-API (single-thread context rule). It hands finished keyframes to the render thread
    // through a newest-wins slot (the harness analogue of ossia::triple_buffer). The render loop, on a
    // STEADY wall-clock tick, advances a phase by real elapsed time and emits a PRECOMPUTED sub-frame
    // (cheap memcpy) — RIFE is run ONCE per adopted keyframe, NOT per tick.
    const int target_render_fps = arg_fps > 0 ? arg_fps : 60;
    const double tick_dt = 1.0 / (double)target_render_fps;
    const double window_s = window_arg;
    const int n_sub = exp > 0 ? (1 << exp) : 1;

    // --- newest-wins keyframe handoff (producer -> render) ---
    std::mutex hb_m;
    std::vector<unsigned char> hb_frame((size_t)n_sub * nbytes, 0); // newest finished SWEEP (producer-rendered)
    int hb_sweep_n = 1;                               // sub-frame count in hb_frame
    bool hb_new = false;                              // a new sweep is waiting to be consumed
    // --- job request (render -> producer), latest-wins ---
    std::atomic<bool> job_req{false};
    std::atomic<bool> stop{false};
    std::atomic<int> worker_diff{0};
    // signal so the producer sleeps instead of spinning
    std::mutex req_m; std::condition_variable req_cv;

    std::thread producer([&]{
      // Producer owns RIFE too (its own handle): it diffuses the keyframe THEN interpolates the sweep,
      // sequentially on this thread. The render thread does NO GPU compute -> no contention starving
      // diffusion. (Mirrors the node: producer publishes the full sweep; render just presents.)
      librediffusion_rife_handle prife = nullptr;
      if(exp > 0) { prife = librediffusion_rife_create(rife_plan, 0);
        if(prife){ librediffusion_rife_set_enabled(prife,1); librediffusion_rife_set_interpolation_exp(prife,exp);} }
      std::vector<unsigned char> out(nbytes, 0), pprev(nbytes, 0), psweep((size_t)n_sub*nbytes, 0);
      bool php = false;  // producer has a previous keyframe
      int t = 1;
      while(!stop.load()) {
        {
          std::unique_lock<std::mutex> lk(req_m);
          req_cv.wait_for(lk, std::chrono::milliseconds(5),
                          [&]{ return job_req.load() || stop.load(); });
        }
        if(stop.load()) break;
        if(!job_req.exchange(false)) continue;
        fill_ref(t++);
        librediffusion_flux2_stream_set_reference(s, ref.data());        // VAE-enc on s->stream (low prio)
        librediffusion_flux2_stream_frame_cached(s, out.data());         // 2-step + decode (~150ms)
        worker_diff++;
        int psn = 1;
        if(exp > 0 && prife && php) {
          int n = 0;
          if(librediffusion_rife_interpolate(prife, pprev.data(), out.data(), H, W, psweep.data(), &n)
                 == LIBREDIFFUSION_SUCCESS && n > 0) psn = n;
          else { std::memcpy(psweep.data(), out.data(), nbytes); psn = 1; }
        } else { std::memcpy(psweep.data(), out.data(), nbytes); psn = 1; }
        pprev = out; php = true;
        { std::lock_guard<std::mutex> lk(hb_m); hb_frame = psweep; hb_sweep_n = psn; hb_new = true; }
        job_req = true;   // self-rearm: run continuously (saturate diffusion at native rate)
      }
      if(prife) librediffusion_rife_destroy(prife);
    });

    // render-thread: SUB-FRAME FIFO, emit ONE per tick (sync's model -> 0% dup, monotonic motion). The
    // pacing mode only bounds how much we buffer (latency vs freshness): 0=smooth(3 sweeps),1=fresh(2),
    // 2=lowlatency(1). If the FIFO drains (producer behind) HOLD the last frame; if over budget, trim front.
    std::deque<std::vector<unsigned char>> frames;
    std::vector<unsigned char> last_emit(nbytes,0); bool have_emit=false;
    double kf_interval = 0.0; double last_kf_t = 0.0;
    double prod_rate = 0.0;       // EMA measured sub-frames/sec
    double drain_credit = 0.0;    // fractional sub-frames owed this tick (even-spread drain)
    double last_tick = 0.0;

    std::vector<unsigned char> prev_emit(nbytes, 0); bool have_prev_emit = false;
    std::vector<double> motion_delta;   // per-frame mean-abs content delta (smoothness-of-MOTION metric)
    motion_delta.reserve((size_t)arg_fps * 16 + 4096);

    double w_start = now_s(), next_tick = w_start;
    job_req = true; req_cv.notify_one();  // kick the first keyframe
    while(now_s() - w_start < window_s) {
      double tnow_tick = now_s();
      double dt = (last_tick > 0.0) ? (tnow_tick - last_tick) : 0.0; last_tick = tnow_tick;
      if(dt > 0.1) dt = 0.1;

      // 1. Drain a fresh SWEEP's sub-frames into the FIFO. Measure kf interval + PRODUCTION RATE.
      std::vector<unsigned char> fresh; int fresh_n = 0; bool adopted=false;
      { std::lock_guard<std::mutex> lk(hb_m);
        if(hb_new) { fresh = hb_frame; fresh_n = hb_sweep_n; hb_new = false; adopted = true; } }
      if(adopted) {
        double tnow = now_s();
        int n = std::max(1, fresh_n);
        if(last_kf_t > 0.0) { double gap = tnow - last_kf_t;
          if(gap > 1e-3 && gap < 5.0) {
            kf_interval = (kf_interval <= 0.0)? gap : 0.8*kf_interval+0.2*gap;
            double inst = (double)n/gap;
            prod_rate = (prod_rate <= 0.0)? inst : 0.8*prod_rate + 0.2*inst; } }
        last_kf_t = tnow;
        int budget_sweeps = (pacing==0)?3:(pacing==1)?2:1;
        size_t max_frames = (size_t)budget_sweeps * n;
        if(fresh_n > 1 && fresh.size() >= (size_t)n*nbytes)
          for(int i=0;i<n;i++) frames.emplace_back(fresh.begin()+(size_t)i*nbytes, fresh.begin()+(size_t)(i+1)*nbytes);
        else frames.emplace_back(fresh.begin(), fresh.begin()+nbytes);
        while(frames.size() > max_frames) frames.pop_front();
      }

      // 2. PRESENT via CREDIT-BASED EVEN-SPREAD drain (advance by measured prod_rate, not 1/tick).
      drain_credit += (prod_rate > 0.0) ? prod_rate*dt : 1.0;
      int to_pop = (int)drain_credit;
      if(to_pop > 0) {
        drain_credit -= (double)to_pop;
        while(to_pop > 0 && !frames.empty()) { last_emit = std::move(frames.front()); frames.pop_front(); have_emit=true; --to_pop; }
        if(to_pop > 0) drain_credit = 0.0;  // starved: drop unmet credit, hold last
      }
      if(have_emit) {
        emit = last_emit;
        if(have_prev_emit) { double s=0; for(size_t k=0;k<nbytes;k++) s+=std::abs((int)emit[k]-(int)prev_emit[k]);
          motion_delta.push_back(s/nbytes); }
        prev_emit = emit; have_prev_emit = true;
        mark_emit();
      }

      // pace to the steady render rate
      next_tick += tick_dt;
      double sleep_s = next_tick - now_s();
      if(sleep_s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
    }
    stop = true; req_cv.notify_one(); producer.join();
    diffusions = worker_diff.load();
    t_start = now_s() - window_s;

    // MOTION-smoothness report (distinct from interval pacing): how steady is the per-frame CONTENT
    // change? Judder (sub-frame repeat-then-jump) shows up as high relative stddev even when intervals
    // are perfect. Smooth interpolation => deltas near-constant => stddev/mean small.
    if(motion_delta.size() > 3) {
      double m=0; for(double v:motion_delta) m+=v; m/=motion_delta.size();
      double var=0; for(double v:motion_delta) var+=(v-m)*(v-m); var/=motion_delta.size();
      double sd=std::sqrt(var);
      // SKIP the first ~90 frames (warmup: producer's first keyframe = empty FIFO -> a long initial hold
      // that is NOT steady-state). Measure clustering on steady-state only.
      size_t warm = std::min<size_t>(90, motion_delta.size()/4);
      int zeros=0; for(size_t i=warm;i<motion_delta.size();i++) if(motion_delta[i] < 0.05) zeros++;
      // CLUSTERING is what the eye sees: isolated repeats (even-spread) are invisible; RUNS of repeats
      // (bunched) are the visible stutter. Measure the max + count of consecutive-dup runs >=2.
      int run=0, maxrun=0, bunched_runs=0;
      for(size_t i=warm;i<motion_delta.size();i++){ double v=motion_delta[i]; if(v<0.05){ run++; if(run>maxrun)maxrun=run; if(run==2)bunched_runs++; } else run=0; }
      std::vector<double> sorted=motion_delta; std::sort(sorted.begin(),sorted.end());
      double mx=sorted.back();
      printf("  --- MOTION smoothness (per-frame content delta) ---\n");
      printf("    mean=%.3f  stddev=%.3f  (rel=%.2f)  max=%.3f  dup-frames(<0.05)=%d/%zu=%.1f%%\n",
             m, sd, (m>1e-6?sd/m:0.0), mx, zeros, motion_delta.size(), 100.0*zeros/motion_delta.size());
      printf("    dup CLUSTERING: max consecutive-dup run=%d  bunched runs(>=2 in a row)=%d  "
             "(even-spread => maxrun 1, 0 bunched = invisible; bunched => visible stutter)\n",
             maxrun, bunched_runs);
      printf("    verdict: %s\n",
             ((m>1e-6?sd/m:9) < 0.5 && 100.0*zeros/motion_delta.size() < 5.0)
               ? "SMOOTH MOTION (steady content advance)"
               : "JUDDER (uneven content advance / dup frames)");
    }
  }
  else { fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 5; }

  double dt = now_s() - t_start;
  printf("\n=== klein FPS harness ===\n");
  printf("mode=%s  pacing=%s  res=%dx%d  exp=%d  arg=%d  wall=%.2fs\n",
         mode.c_str(), pacing_arg.c_str(), W, H, exp, arg_fps, dt);
  printf("  diffusions=%d  -> diffusion fps = %.2f\n", diffusions, diffusions / dt);
  printf("  emitted=%d     -> avg DISPLAY fps = %.2f\n", emitted, emitted / dt);

  report_pacing(emit_ts);

  if(CLIPDIR && !clip.empty()) {
    for(size_t i = 0; i < clip.size(); i++) {
      char p[1024]; std::snprintf(p, sizeof(p), "%s/frame_%05zu.ppm", CLIPDIR, i + 1);
      std::ofstream o(p, std::ios::binary); o << "P6\n" << W << " " << H << "\n255\n";
      const auto& f = clip[i];
      for(size_t k = 0; k < (size_t)W * H; k++) { o.put(f[k*4]); o.put(f[k*4+1]); o.put(f[k*4+2]); }
    }
    printf("  CLIP: wrote %zu frames to %s\n", clip.size(), CLIPDIR);
  }

  if(rife) librediffusion_rife_destroy(rife);
  librediffusion_flux2_stream_destroy(s);
  return 0;
}
