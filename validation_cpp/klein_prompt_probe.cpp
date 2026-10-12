/* klein_prompt_probe: does the PROMPT actually change the output?
 * Fixed reference + fixed seed; run frame_cached under several prompts; report mean abs pixel delta
 * between each pair. If deltas ~0, the prompt has no effect (the bug the user reports). */
#include "librediffusion_c.h"
#include "qwen_tokenizer_c.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

static std::vector<float> load_bin_f32(const char* p, size_t n){
  FILE* f=fopen(p,"rb"); std::vector<float> v(n,0.f);
  if(f){ size_t r=fread(v.data(),4,n,f); (void)r; fclose(f);} return v;
}
static double mad(const std::vector<unsigned char>&a,const std::vector<unsigned char>&b){
  double s=0; size_t n=std::min(a.size(),b.size());
  for(size_t i=0;i<n;i++) s+=std::abs((int)a[i]-(int)b[i]); return s/std::max<size_t>(1,n);
}
int main(int argc,char**argv){
  if(argc<9){ fprintf(stderr,"usage: %s <transformer> <qwen> <vae_dec> <vae_enc> <tok.json> <bn_mean> <bn_std> [W=320] [H=576]\n",argv[0]); return 1; }
  const char* transformer=argv[1]; const char* qwen=argv[2]; const char* vae_dec=argv[3];
  const char* vae_enc=argv[4]; const char* tok=argv[5]; const char* bnm=argv[6]; const char* bns=argv[7];
  const int W=argc>8?atoi(argv[8]):320; const int H=argc>9?atoi(argv[9]):576;
  const int Th=H/16, Tw=W/16; const size_t nbytes=(size_t)W*H*4;
  if(qwen_tokenizer_load(tok)!=0){ fprintf(stderr,"tok load failed\n"); return 2; }
  auto* s=librediffusion_flux2_stream_create(transformer,qwen,vae_dec,vae_enc,tok,Th,Tw,52ull,0);
  if(!s){ fprintf(stderr,"create failed\n"); return 3; }
  auto bm=load_bin_f32(bnm,128), bs=load_bin_f32(bns,128);
  librediffusion_flux2_stream_set_bn(s,bm.data(),bs.data());
  librediffusion_flux2_stream_set_steps(s,2);

  // fixed reference frame (a gradient)
  std::vector<unsigned char> ref(nbytes,0);
  for(int y=0;y<H;y++)for(int x=0;x<W;x++){long i=((long)y*W+x)*4;
    ref[i]=(unsigned char)(x&255); ref[i+1]=(unsigned char)(y&255); ref[i+2]=128; ref[i+3]=255;}
  librediffusion_flux2_stream_set_reference(s,ref.data());

  const char* prompts[]={
    "turn this into an oil painting, masterpiece",
    "a cyberpunk neon city at night, rain, dramatic",
    "a watercolor sketch of mountains, soft pastel",
    "make everything bright red, abstract fire"
  };
  const int NP=4;
  std::vector<std::vector<unsigned char>> outs(NP, std::vector<unsigned char>(nbytes,0));
  for(int p=0;p<NP;p++){
    int rc=librediffusion_flux2_stream_set_prompt(s,prompts[p]);
    printf("set_prompt[%d] returned %d (\"%s\")\n", p, rc, prompts[p]);
    librediffusion_flux2_stream_frame_cached(s,outs[p].data());
    // checksum
    unsigned long long sum=0; for(unsigned char c:outs[p]) sum+=c;
    printf("  output[%d] bytesum=%llu  first8=%d,%d,%d,%d,%d,%d,%d,%d\n", p, sum,
      outs[p][0],outs[p][1],outs[p][2],outs[p][3],outs[p][4],outs[p][5],outs[p][6],outs[p][7]);
  }
  printf("\n--- mean abs pixel delta between prompt outputs (0 = prompt has NO effect) ---\n");
  for(int a=0;a<NP;a++)for(int b=a+1;b<NP;b++)
    printf("  MAD(prompt%d, prompt%d) = %.3f\n", a, b, mad(outs[a],outs[b]));
  // re-run prompt0 again to confirm determinism (should be MAD 0 vs first prompt0)
  std::vector<unsigned char> rerun(nbytes,0);
  librediffusion_flux2_stream_set_prompt(s,prompts[0]);
  librediffusion_flux2_stream_frame_cached(s,rerun.data());
  printf("\n  MAD(prompt0, prompt0-rerun) = %.3f  (should be ~0: determinism check)\n", mad(outs[0],rerun));
  librediffusion_flux2_stream_destroy(s);
  return 0;
}
