// sensevoice-server — OpenAI-compatible STT server for SenseVoiceSmall on ggml.
//
//   sensevoice-server -m model/sensevoice-small-q8.gguf -vad model/fsmn-vad.gguf [host] [port]
//
// Endpoints
//   POST /v1/audio/transcriptions           OpenAI file-transcription API (multipart form-data)
//   GET  /v1/models                         OpenAI-compatible model list
//   GET  /health                            liveness
//   WS   /v1/realtime?intent=transcription  OpenAI-compatible realtime transcription session
//
// The engine (80-mel fbank + LFR, SAN-M encoder, CTC head, SentencePiece detok) is
// duplicated verbatim from funasr-sensevoice.cpp so the validated CLI is untouched.
// Streaming uses an incremental copy of the FSMN-VAD state machine (funasr_vad.h)
// to end-point utterances and expose live speech windows for partial transcripts.
//
// Transport: cpp-httplib (vendored, funasr-common/httplib.h). No external deps.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <ctime>

// WebSocket socket read timeout, seconds. cpp-httplib applies
// CPPHTTPLIB_WEBSOCKET_READ_TIMEOUT_SECOND to every upgraded connection; routing it
// through a runtime variable makes it follow --ws-idle-timeout-s (+2 s backstop after the
// close frame) instead of holding a connection slot for 300 s.
static time_t g_ws_idle_timeout_s = 60;
#define CPPHTTPLIB_WEBSOCKET_READ_TIMEOUT_SECOND (g_ws_idle_timeout_s)

#define FUNASR_AUDIO_IMPLEMENTATION
#include "funasr_audio.h"
#include "funasr_vad.h"
#include "httplib.h"
#include "server_utils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static const float LN_EPS = 1e-5f;

using funasr_server::base64_decode;
using funasr_server::format_subtitles;
using funasr_server::format_verbose_segments;
using funasr_server::json_field;
using funasr_server::json_str;
using funasr_server::TranscriptSegment;

static long long now_ms(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ===========================================================================
// SenseVoice engine (duplicated from funasr-sensevoice.cpp)
// ===========================================================================
static const int FS=16000,WINLEN=400,SHIFT=160,NFFT=512,NMEL=80,LFR_M=7,LFR_N=6;
static const float PREEMPH=0.97f,LOWF=20.0f,HIGHF=8000.0f;
// RMS below this (16k f32, normalized) is treated as silence so an idle connection
// does not keep the VAD NN busy re-scoring quiet chunks.
static const float QUIET_RMS = 1e-3f;
static inline float melf(float f){return 1127.0f*logf(1.0f+f/700.0f);}
static void fftc(std::vector<float>&re,std::vector<float>&im,int n){
  for(int i=1,j=0;i<n;i++){int b=n>>1;for(;j&b;b>>=1)j^=b;j^=b;if(i<j){std::swap(re[i],re[j]);std::swap(im[i],im[j]);}}
  for(int len=2;len<=n;len<<=1){double a=-2.0*M_PI/len;float wr=cosf(a),wi=sinf(a);
    for(int i=0;i<n;i+=len){float cr=1,ci=0;for(int k=0;k<len/2;k++){float ur=re[i+k],ui=im[i+k];
      float vr=re[i+k+len/2]*cr-im[i+k+len/2]*ci,vi=re[i+k+len/2]*ci+im[i+k+len/2]*cr;
      re[i+k]=ur+vr;im[i+k]=ui+vi;re[i+k+len/2]=ur-vr;im[i+k+len/2]=ui-vi;float nc=cr*wr-ci*wi;ci=cr*wi+ci*wr;cr=nc;}}}
}
static std::vector<float> compute_fbank(std::vector<float> wav,int&T_out){
  for(auto&v:wav)v*=32768.0f; std::vector<float> win(WINLEN);
  for(int i=0;i<WINLEN;i++)win[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
  const int NBIN=NFFT/2+1; float bw=(float)FS/NFFT,ml=melf(LOWF),mh=melf(HIGHF),dm=(mh-ml)/(NMEL+1);
  std::vector<std::vector<float>> fb(NMEL,std::vector<float>(NBIN,0.0f));
  for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
    for(int k=0;k<NBIN;k++){float mf=melf(bw*k); if(mf>L&&mf<R)fb[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
  int N=wav.size(),T=(N-WINLEN)/SHIFT+1; if(T<1)T=0; std::vector<std::vector<float>> feat(T,std::vector<float>(NMEL));
  std::vector<float> re(NFFT),im(NFFT),fr(WINLEN); const float fl=1.1920929e-07f;
  for(int t=0;t<T;t++){const float*s=wav.data()+t*SHIFT; double mn=0; for(int i=0;i<WINLEN;i++)mn+=s[i]; mn/=WINLEN;
    for(int i=0;i<WINLEN;i++)fr[i]=s[i]-(float)mn; for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1]; fr[0]-=PREEMPH*fr[0];
    for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win[i]:0.0f;im[i]=0.0f;} fftc(re,im,NFFT);
    for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NBIN;k++)if(fb[m][k]>0)e+=fb[m][k]*(re[k]*re[k]+im[k]*im[k]); feat[t][m]=logf(e>fl?e:fl);}}
  const int pad=(LFR_M-1)/2; int Tl=(T+LFR_N-1)/LFR_N;
  if(Tl<1) { T_out=0; return {}; }
  std::vector<std::vector<float>> pd; pd.reserve(T+pad+LFR_M);
  for(int i=0;i<pad;i++)pd.push_back(feat[0]); for(int t=0;t<T;t++)pd.push_back(feat[t]);
  while((int)pd.size()<(Tl-1)*LFR_N+LFR_M)pd.push_back(feat[T-1]);
  int D=LFR_M*NMEL; std::vector<float> out((size_t)Tl*D);
  for(int i=0;i<Tl;i++)for(int j=0;j<LFR_M;j++)memcpy(&out[(size_t)i*D+j*NMEL],pd[i*LFR_N+j].data(),NMEL*sizeof(float));
  T_out=Tl; return out;
}

struct cfg { int d_model=512,n_head=4,num_blocks=50,tp_blocks=20,kernel=11,vocab=25055,blank=0; };
struct model { cfg c; ggml_context*ctx_w=nullptr; std::map<std::string,ggml_tensor*> t;
  ggml_tensor* g(const std::string&n){auto it=t.find(n); if(it==t.end()){fprintf(stderr,"missing %s\n",n.c_str());exit(1);} return it->second;} };

static ggml_tensor* lin(ggml_context*c,ggml_tensor*w,ggml_tensor*b,ggml_tensor*x){auto y=ggml_mul_mat(c,w,x);return b?ggml_add(c,y,b):y;}
static ggml_tensor* lnorm(ggml_context*c,ggml_tensor*x,ggml_tensor*g,ggml_tensor*b){return ggml_add(c,ggml_mul(c,ggml_norm(c,x,LN_EPS),g),b);}
static ggml_tensor* sanm_attn(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T){
  const int D=m.c.d_model,H=m.c.n_head,dk=D/H,K=m.c.kernel;
  ggml_tensor*qkv=lin(c,m.g(p+"linear_q_k_v.weight"),m.g(p+"linear_q_k_v.bias"),x); size_t nb1=qkv->nb[1];
  ggml_tensor*q=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,0));
  ggml_tensor*k=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)D*sizeof(float)));
  ggml_tensor*v=ggml_cont(c,ggml_view_2d(c,qkv,D,T,nb1,(size_t)2*D*sizeof(float)));
  const int pad=(K-1)/2; ggml_tensor*fk=m.g(p+"fsmn_block.weight");
  ggml_tensor*vp=ggml_pad_ext(c,v,0,0,pad,pad,0,0,0,0); ggml_tensor*fsmn=v;
  for(int j=0;j<K;j++){auto sl=ggml_view_2d(c,vp,D,T,vp->nb[1],(size_t)j*vp->nb[1]);
    auto wj=ggml_view_1d(c,fk,D,(size_t)j*fk->nb[1]); fsmn=ggml_add(c,fsmn,ggml_mul(c,ggml_cont(c,sl),wj));}
  q=ggml_permute(c,ggml_reshape_3d(c,q,dk,H,T),0,2,1,3); k=ggml_permute(c,ggml_reshape_3d(c,k,dk,H,T),0,2,1,3);
  ggml_tensor*vh=ggml_cont(c,ggml_permute(c,ggml_reshape_3d(c,v,dk,H,T),1,2,0,3));
  ggml_tensor*kq=ggml_soft_max(c,ggml_scale(c,ggml_mul_mat(c,k,q),1.0f/sqrtf((float)dk)));
  ggml_tensor*o=ggml_cont_2d(c,ggml_permute(c,ggml_mul_mat(c,vh,kq),0,2,1,3),D,T);
  return ggml_add(c,lin(c,m.g(p+"linear_out.weight"),m.g(p+"linear_out.bias"),o),fsmn);
}
static ggml_tensor* sanm_layer(ggml_context*c,model&m,const std::string&p,ggml_tensor*x,int T,bool res){
  auto r=x; auto h=lnorm(c,x,m.g(p+"norm1.weight"),m.g(p+"norm1.bias"));
  auto sa=sanm_attn(c,m,p+"self_attn.",h,T); x=res?ggml_add(c,r,sa):sa; r=x;
  h=lnorm(c,x,m.g(p+"norm2.weight"),m.g(p+"norm2.bias"));
  h=lin(c,m.g(p+"feed_forward.w_1.weight"),m.g(p+"feed_forward.w_1.bias"),h); h=ggml_relu(c,h);
  h=lin(c,m.g(p+"feed_forward.w_2.weight"),m.g(p+"feed_forward.w_2.bias"),h); return ggml_add(c,r,h);
}
static void add_posenc(std::vector<float>&x,int T,int depth){
  double inc=log(10000.0)/(depth/2.0-1.0);
  for(int t=0;t<T;t++){double pos=t+1;for(int i=0;i<depth/2;i++){double its=exp(i*-inc),st=pos*its;
    x[(size_t)t*depth+i]+=(float)sin(st);x[(size_t)t*depth+depth/2+i]+=(float)cos(st);}}
}

struct sv_out { std::string text; std::vector<std::string> tags; std::string language; bool aborted=false; };
static std::string sv_trim(const std::string&s){size_t a=s.find_first_not_of(' ');if(a==std::string::npos)return "";size_t b=s.find_last_not_of(' ');return s.substr(a,b-a+1);}
static bool sv_is_lang(const std::string&t){
    static const std::vector<std::string> L={
        "<|auto|>","<|zh|>","<|en|>","<|yue|>","<|ja|>","<|ko|>","<|nospeech|>",
        "<|pt|>","<|ru|>","<|es|>","<|it|>","<|fr|>","<|de|>","<|nl|>","<|pl|>","<|tr|>",
        "<|ar|>","<|hi|>","<|vi|>","<|th|>","<|id|>","<|ms|>","<|fa|>"};
    return std::find(L.begin(),L.end(),t)!=L.end();
}
static sv_out detok_sv_full(const std::vector<int>&ids,const std::vector<std::string>&vocab,bool keep_tags){
    sv_out o; std::string s;
    for(int id:ids){ if(id<0||id>=(int)vocab.size())continue; const std::string&p=vocab[id];
        if(p.size()>=2 && p[0]=='<' && p[1]=='|'){
            if(sv_is_lang(p) && o.language.empty()) o.language = p.substr(2, p.size()-4);
            o.tags.push_back(p.substr(2, p.size()-4));
            if(keep_tags) s+=p;
            continue;
        }
        s+=p;
    }
    const std::string lb="\xe2\x96\x81"; size_t pp; while((pp=s.find(lb))!=std::string::npos)s.replace(pp,3," ");
    o.text = sv_trim(s);
    return o;
}

struct SenseVoice {
    model m; int F=560,V=-1,nq=0,nthreads=8;
    std::vector<int> qtok; std::vector<std::string> vocab;
    bool emit_ids=false;
    std::mutex dec;   // serializes the ggml forward (single-CPU edge target)
    int ngl=0;                              // >0 -> run the whole model on CUDA device 0
    std::vector<float> emb_qtok;            // host copy of the query-token embedding rows
#ifdef GGML_USE_CUDA
    ggml_backend_t gpu_be=nullptr;
    ggml_backend_buffer_t gpu_wbuf=nullptr; // weights GPU buffer
#endif

    ~SenseVoice(){
#ifdef GGML_USE_CUDA
        if(gpu_wbuf) ggml_backend_buffer_free(gpu_wbuf);
        if(gpu_be)   ggml_backend_free(gpu_be);
#endif
    }

    bool load(const char* path){
        gguf_init_params gp={false,&m.ctx_w};
        gguf_context*gg=gguf_init_from_file(path,gp);
        if(!gg){fprintf(stderr,"sensevoice: load gguf failed: %s\n",path);return false;}
        auto rd=[&](const char*k,int d){int i=gguf_find_key(gg,k);return i<0?d:(int)gguf_get_val_u32(gg,i);};
        m.c.d_model=rd("sv.output_size",512); m.c.n_head=rd("sv.attention_heads",4);
        m.c.num_blocks=rd("sv.num_blocks",50); m.c.tp_blocks=rd("sv.tp_blocks",20);
        m.c.kernel=rd("sv.kernel_size",11); m.c.vocab=rd("sv.vocab_size",25055); m.c.blank=rd("sv.blank_id",0);
        int qi=gguf_find_key(gg,"sv.query_tokens"); nq=qi<0?0:(int)gguf_get_arr_n(gg,qi);
        qtok.resize(nq);
        if(nq>0) memcpy(qtok.data(),gguf_get_arr_data(gg,qi),(size_t)nq*sizeof(int32_t));
        int ki=gguf_find_key(gg,"sv.vocab");
        if(ki>=0){ int nv=gguf_get_arr_n(gg,ki); vocab.resize(nv); for(int i=0;i<nv;i++) vocab[i]=gguf_get_arr_str(gg,ki,i)?gguf_get_arr_str(gg,ki,i):""; }
        for(int i=0;i<gguf_get_n_tensors(gg);i++){const char*nm=gguf_get_tensor_name(gg,i);m.t[nm]=ggml_get_tensor(m.ctx_w,nm);}
        gguf_free(gg);
        emb_qtok.resize((size_t)nq*F);
        if(nq>0){ const float*emb=(const float*)m.g("embed.weight")->data;
            for(int i=0;i<nq;i++) memcpy(emb_qtok.data()+(size_t)i*F, emb+(size_t)qtok[i]*F, (size_t)F*sizeof(float)); }
        V=m.c.vocab; emit_ids = vocab.empty();
        return V>0 && m.ctx_w!=nullptr;
    }

    // Offload the model to CUDA device 0. This runtime builds raw ggml graphs (no llama
    // model loader), so there is no per-layer split: any n_gpu_layers>0 moves every weight
    // tensor into a device buffer and runs the whole forward on the GPU. Falls back to CPU
    // when CUDA is unavailable or the binary was built without GGML_CUDA.
    bool init_gpu(int n_gpu_layers){
        ngl = n_gpu_layers>0 ? n_gpu_layers : 0;
        if(ngl==0) return true;
#ifdef GGML_USE_CUDA
        if(ggml_backend_cuda_get_device_count() < 1){
            fprintf(stderr,"sensevoice: no CUDA device found; running on CPU\n");
            ngl=0; return true;
        }
        ggml_backend_t be=ggml_backend_cuda_init(0);
        if(!be){ fprintf(stderr,"sensevoice: CUDA backend init failed; running on CPU\n"); ngl=0; return true; }
        ggml_backend_buffer_type_t buft=ggml_backend_cuda_buffer_type(0);
        size_t align=ggml_backend_buft_get_alignment(buft), total=0;
        for(auto&kv:m.t) if(!kv.second->view_src)
            total += GGML_PAD(ggml_backend_buft_get_alloc_size(buft,kv.second),align);
        ggml_backend_buffer_t wbuf=ggml_backend_buft_alloc_buffer(buft,total);
        if(!wbuf){
            fprintf(stderr,"sensevoice: cannot allocate %.1f MiB on GPU; running on CPU\n",(double)total/(1024.0*1024.0));
            ggml_backend_free(be); ngl=0; return true;
        }
        size_t off=0;
        for(auto&kv:m.t){
            ggml_tensor*tensor=kv.second;
            if(tensor->view_src) continue;
            void* cpu=tensor->data;
            size_t sz=GGML_PAD(ggml_backend_buft_get_alloc_size(buft,tensor),align);
            tensor->data=nullptr;   // ggml_backend_tensor_alloc requires data==NULL
            if(ggml_backend_tensor_alloc(wbuf,tensor,(char*)ggml_backend_buffer_get_base(wbuf)+off)!=GGML_STATUS_SUCCESS){
                // tensors already moved point into wbuf; cannot safely continue on CPU
                fprintf(stderr,"sensevoice: GPU weight upload failed\n");
                ggml_backend_buffer_free(wbuf); ggml_backend_free(be);
                return false;
            }
            ggml_backend_tensor_set(tensor,cpu,0,ggml_nbytes(tensor));
            off += sz;
        }
        gpu_be=be; gpu_wbuf=wbuf;
        size_t mb=(total+1024*1024-1)/(1024*1024);
        fprintf(stderr,"[server] offloaded %zu/%zu tensors (%.1f MiB) to CUDA device 0\n",
                m.t.size(),m.t.size(),(double)mb);
        return true;
#else
        fprintf(stderr,"sensevoice: -ngl requires a CUDA build (cmake -DGGML_CUDA=ON); running on CPU\n");
        ngl=0;
        return true;
#endif
    }

    // abort_cb (optional, CPU backend only) is polled by ggml between graph nodes; when it
    // returns true the forward stops early and an empty result with aborted=true is returned.
    sv_out transcribe_fb(const std::vector<float>& fb,int T,bool keep_tags,
                         ggml_abort_callback abort_cb=nullptr,void* abort_data=nullptr){
        std::lock_guard<std::mutex> lk(dec);
        if(abort_cb && abort_cb(abort_data)){ sv_out o; o.aborted=true; return o; }
        const int D=m.c.d_model;
        int N=nq+T; std::vector<float> inp((size_t)N*F);
        for(int i=0;i<nq;i++) memcpy(&inp[(size_t)i*F], emb_qtok.data()+(size_t)i*F, (size_t)F*sizeof(float));
        memcpy(&inp[(size_t)nq*F], fb.data(), (size_t)T*F*sizeof(float));
        float sc=sqrtf((float)D); for(auto&v:inp)v*=sc; add_posenc(inp,N,F);
        bool is_gpu=false;
#ifdef GGML_USE_CUDA
        is_gpu = gpu_be!=nullptr;
        ggml_backend_t be = is_gpu ? gpu_be : ggml_backend_cpu_init();
        ggml_backend_buffer_type_t buft = is_gpu ? ggml_backend_cuda_buffer_type(0)
                                                 : ggml_backend_cpu_buffer_type();
#else
        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
#endif
        ggml_init_params cp={(size_t)1024*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
        ggml_tensor*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,F,N); ggml_set_input(x);
        ggml_tensor*h=sanm_layer(c,m,"encoder.encoders0.0.",x,N,false);
        for(int i=0;i<m.c.num_blocks-1;i++) h=sanm_layer(c,m,"encoder.encoders."+std::to_string(i)+".",h,N,true);
        h=lnorm(c,h,m.g("encoder.after_norm.weight"),m.g("encoder.after_norm.bias"));
        for(int i=0;i<m.c.tp_blocks;i++) h=sanm_layer(c,m,"encoder.tp_encoders."+std::to_string(i)+".",h,N,true);
        h=lnorm(c,h,m.g("encoder.tp_norm.weight"),m.g("encoder.tp_norm.bias"));
        ggml_tensor*logits=lin(c,m.g("ctc.ctc_lo.weight"),m.g("ctc.ctc_lo.bias"),h);  // [V, N]
        ggml_set_output(logits);
        ggml_cgraph*gf=ggml_new_graph_custom(c,32768,false); ggml_build_forward_expand(gf,logits);
        ggml_gallocr_t ga=ggml_gallocr_new(buft); ggml_gallocr_alloc_graph(ga,gf);
        ggml_backend_tensor_set(x,inp.data(),0,ggml_nbytes(x));
        if(!is_gpu){ ggml_backend_cpu_set_n_threads(be,nthreads); ggml_backend_cpu_set_abort_callback(be,abort_cb,abort_data); }
        ggml_status st=ggml_backend_graph_compute(be,gf);
        if(st==GGML_STATUS_ABORTED){
            ggml_gallocr_free(ga); ggml_free(c);
            if(!is_gpu) ggml_backend_free(be);
            sv_out o; o.aborted=true; return o;
        }
        if(st!=GGML_STATUS_SUCCESS){fprintf(stderr,"sensevoice: compute failed\n");}
        std::vector<float> lg((size_t)V*N); ggml_backend_tensor_get(logits,lg.data(),0,ggml_nbytes(logits));
        std::vector<int> seg_ids; int prev=-1;
        for(int n=0;n<N;n++){ const float*col=&lg[(size_t)n*V]; int am=0; float best=col[0];
            for(int v=1;v<V;v++) if(col[v]>best){best=col[v];am=v;}
            if(am!=prev && am!=m.c.blank) seg_ids.push_back(am); prev=am; }
        ggml_gallocr_free(ga); ggml_free(c);
        if(!is_gpu) ggml_backend_free(be);
        return emit_ids ? sv_out{} : detok_sv_full(seg_ids,vocab,keep_tags);
    }

    sv_out transcribe_wav(const std::vector<float>& wav,bool keep_tags,
                          ggml_abort_callback abort_cb=nullptr,void* abort_data=nullptr){
        if((int)wav.size()<WINLEN) return sv_out{};
        int T=0; auto fb=compute_fbank(wav,T);
        if(T<1) return sv_out{};
        return transcribe_fb(fb,T,keep_tags,abort_cb,abort_data);
    }
};

// ---------------------------------------------------------------------------
// Audio decode from memory (uploads) — miniaudio compiled in this TU.
// ---------------------------------------------------------------------------
enum class AudioDecodeResult { Ok, Invalid, TooLarge };
static AudioDecodeResult decode_audio_mem(const void* data, size_t len, size_t max_samples, std::vector<float>& out){
    if(!data||!len) return AudioDecodeResult::Invalid;
    ma_decoder_config cfg = ma_decoder_config_init(ma_format_f32, 1, 16000);
    ma_decoder dec;
    if(ma_decoder_init_memory(data, len, &cfg, &dec)!=MA_SUCCESS){fprintf(stderr,"audio: failed to decode upload\n");return AudioDecodeResult::Invalid;}
    out.clear();
    ma_uint64 nframes = 0;
    if(ma_decoder_get_length_in_pcm_frames(&dec,&nframes)==MA_SUCCESS && nframes>0){
        if(nframes>max_samples){ ma_decoder_uninit(&dec); return AudioDecodeResult::TooLarge; }
        out.resize((size_t)nframes);
        ma_uint64 got=0;
        ma_result r=ma_decoder_read_pcm_frames(&dec,out.data(),nframes,&got);
        if(r!=MA_SUCCESS && r!=MA_AT_END){ma_decoder_uninit(&dec);return AudioDecodeResult::Invalid;}
        out.resize((size_t)got);
    } else {
        std::vector<float> buf(16000);
        for(;;){
            ma_uint64 got=0;
            ma_result r=ma_decoder_read_pcm_frames(&dec,buf.data(),buf.size(),&got);
            if(r!=MA_SUCCESS && r!=MA_AT_END){ out.clear(); break; }
            if(got==0) break;
            if(got>max_samples || out.size()>max_samples-(size_t)got){
                ma_decoder_uninit(&dec);
                out.clear();
                return AudioDecodeResult::TooLarge;
            }
            out.insert(out.end(),buf.begin(),buf.begin()+got);
            if(r==MA_AT_END && got<buf.size()) break;
        }
    }
    ma_decoder_uninit(&dec);
    return out.empty() ? AudioDecodeResult::Invalid : AudioDecodeResult::Ok;
}

// ===========================================================================
// Incremental FSMN-VAD (streaming state machine, faithful to funasr_vad.h)
// ===========================================================================
enum VadEvType { VAD_SPEECH_START, VAD_SPEECH_STOP, VAD_SEGMENT };
struct VadEvent { VadEvType type; int start_ms, end_ms; };

struct VadStream {
    ggml_context* ctx_w = nullptr;
    std::map<std::string,ggml_tensor*> t;
    ggml_tensor* g(const std::string&n){auto it=t.find(n); return it==t.end()?nullptr:it->second;}

    int idim=400,pd=128,nl=4,lorder=20,od=248,lm=5,ln=1;
    int nthreads=8, max_seg_ms=30000;
    // Soft split (streaming only): once the open segment is longer than soft_seg frames,
    // end it at the first pause of soft_sil frames instead of waiting for the full
    // end-of-turn silence (~1.85 s) or the hard --vad-maxseg cut. 0 disables.
    int soft_seg=0, soft_sil=0;

    std::vector<float> wav;               // accumulated 16k mono f32 samples
    int stepped=0;                        // frames already pushed through the SM

    // E2EVadModel state machine (copied from funasr_vad.h, driven incrementally)
    const int FR=10, win=20, s2s=15, sp2s=15, lookahead_end=10;
    int max_seg = 3000;                    // frames
    const int start_lookback = 40;         // win + 200/FR
    const int CHUNK = 6000;                // 60000/FR
    int acc=0, insp=0, max_end_sil=0, end_lookback=0;
    std::vector<int> wbuf; int wpos=0,wsum=0,pre=0;
    int st_sm=0, cstart=-1, csil=0, prev_end=0;
    bool ok=false;

    // incremental feature caches (10ms frames; NMEL floats/frame for feat_, od/frame for sil_)
    std::vector<float> feat_;                 // cached fbank rows
    std::vector<float> sil_;                  // cached softmax score rows
    std::vector<std::vector<float>> fb_;      // mel filterbank (lazy)
    std::vector<float> win_;                  // hamming window (lazy)

    bool load(const char* path){
        gguf_init_params ip={false,&ctx_w};
        gguf_context*gg=gguf_init_from_file(path,ip);
        if(!gg){fprintf(stderr,"vad: cannot load %s\n",path);return false;}
        auto rd=[&](const char*k,int d){int i=gguf_find_key(gg,k);return i<0?d:(int)gguf_get_val_u32(gg,i);};
        idim=rd("vad.input_dim",400); pd=rd("vad.proj_dim",128); nl=rd("vad.fsmn_layers",4);
        lorder=rd("vad.lorder",20); od=rd("vad.output_dim",248); lm=rd("vad.lfr_m",5); ln=rd("vad.lfr_n",1);
        for(int i=0;i<gguf_get_n_tensors(gg);i++){const char*nm=gguf_get_tensor_name(gg,i); t[nm]=ggml_get_tensor(ctx_w,nm);}
        gguf_free(gg);
        ok = g("cmvn.shift")&&g("cmvn.scale")
            &&g("encoder.in_linear1.linear.weight")&&g("encoder.in_linear2.linear.weight")
            &&g("encoder.out_linear1.linear.weight")&&g("encoder.out_linear2.linear.weight");
        for(int i=0;i<nl&&ok;i++){std::string p="encoder.fsmn."+std::to_string(i)+".";
            ok=g(p+"linear.linear.weight")&&g(p+"fsmn_block.conv_left.weight")&&g(p+"affine.linear.weight");}
        if(!ok){ fprintf(stderr,"vad: gguf missing required tensors\n"); }
        max_seg=(max_seg_ms>0?max_seg_ms:60000)/FR; if(max_seg<1)max_seg=1;
        wbuf.assign(win,0);
        recompute();
        return ok;
    }

    void recompute(){
        int s; if(acc<=10000)s=2000; else if(acc<=20000)s=1000; else if(acc<=30000)s=800;
        else if(acc<=40000)s=600; else if(acc<=50000)s=400; else if(acc<=60000)s=200; else s=100;
        int ms=s-150; if(ms<0)ms=0; max_end_sil=ms/FR;
        end_lookback=max_end_sil-lookahead_end-1; if(end_lookback<0)end_lookback=0;
    }
    void reset_sm(){
        std::fill(wbuf.begin(),wbuf.end(),0); wpos=0; wsum=0; pre=0; csil=0; st_sm=0; cstart=-1; acc=0; insp=0;
    }

    void feed(const float* s, size_t n){ wav.insert(wav.end(), s, s+n); }

    void ensure_fb(){
        if(!fb_.empty()) return;
        const int NBIN=NFFT/2+1; float bw=(float)FS/NFFT,ml=melf(LOWF),mh=melf(HIGHF),dm=(mh-ml)/(NMEL+1);
        fb_.assign(NMEL,std::vector<float>(NBIN,0.0f));
        for(int m=0;m<NMEL;m++){float L=ml+m*dm,C=ml+(m+1)*dm,R=ml+(m+2)*dm;
            for(int k=0;k<NBIN;k++){float mf=melf(bw*k); if(mf>L&&mf<R)fb_[m][k]=mf<=C?(mf-L)/(C-L):(R-mf)/(R-C);}}
        win_.assign(WINLEN,0.0f);
        for(int i=0;i<WINLEN;i++)win_[i]=0.54f-0.46f*cosf(2.0f*M_PI*i/(WINLEN-1));
    }

    // Compute the fbank row for frame t (10ms, t*SHIFT) and append it to feat_.
    // Each frame's FFT is independent, so only new tail frames need recomputing.
    void compute_fbank_frame(int t){
        ensure_fb();
        const float* s=wav.data()+(size_t)t*SHIFT;
        const int NBIN=NFFT/2+1; const float fl=1.1920929e-07f;
        std::vector<float> fr(WINLEN);
        double mn=0; for(int i=0;i<WINLEN;i++)mn+=s[i]*32768.0f; mn/=WINLEN;
        for(int i=0;i<WINLEN;i++)fr[i]=s[i]*32768.0f-(float)mn;
        for(int i=WINLEN-1;i>0;i--)fr[i]-=PREEMPH*fr[i-1]; fr[0]-=PREEMPH*fr[0];
        std::vector<float> re(NFFT),im(NFFT);
        for(int i=0;i<NFFT;i++){re[i]=i<WINLEN?fr[i]*win_[i]:0.0f;im[i]=0.0f;}
        fftc(re,im,NFFT);
        for(int m=0;m<NMEL;m++){float e=0;for(int k=0;k<NBIN;k++)if(fb_[m][k]>0)e+=fb_[m][k]*(re[k]*re[k]+im[k]*im[k]);feat_.push_back(logf(e>fl?e:fl));}
    }

    // Incremental VAD: compute fbank/LFR/NN only for frames [stepped, T) and append
    // the new softmax rows to sil_. Returns false when no new frames can be scored.
    bool frames_are_quiet(int b0, int b1) const {
        double e=0; size_t cnt=0;
        for(int t=b0;t<b1;t++){
            const float* s=wav.data()+(size_t)t*SHIFT;
            for(int i=0;i<WINLEN;i++){ float v=s[i]; e += (double)v*v; }
            cnt += WINLEN;
        }
        return cnt==0 || sqrt(e/(double)cnt) < QUIET_RMS;
    }

    bool forward_new_scores(int& T){
        int new_T=((int)wav.size()-WINLEN)/SHIFT+1; if(new_T<1)new_T=0;
        T=new_T;
        if(new_T<=stepped) return false;
        int f_n=(int)feat_.size()/NMEL;
        if(new_T>f_n){ feat_.reserve((size_t)new_T*NMEL); for(int t=f_n;t<new_T;t++)compute_fbank_frame(t); }
        int b0=stepped,b1=new_T,nb=b1-b0;
        // Idle gate: while no segment is open, quiet frames become plain silence
        // votes in the score cache without spawning ggml threads. An open segment
        // (or real speech) still runs the NN on every frame, so speech onset and
        // offset detection are unchanged; a quiet, connected-but-idle client just
        // stops burning a full core per poll.
        if(st_sm==0 && nb>0 && frames_are_quiet(b0,b1)){
            size_t old=sil_.size(); sil_.resize(old+(size_t)od*nb);
            for(int i=0;i<nb;i++){ float* r=&sil_[old+(size_t)i*od]; memset(r,0,od*sizeof(float)); r[0]=1.0f; }
            stepped = new_T;
            return false;
        }
        // LFR rows [b0,b1): row i = concat(feat[i-2..i+2]) with the exact tail/head clamp of lfr(m=5,n=1).
        std::vector<float> feats((size_t)nb*idim);
        for(int i=b0;i<b1;i++){
            float* out=&feats[(size_t)(i-b0)*idim];
            int o=0;
            for(int j=0;j<lm;j++){
                int idx=i+j-(lm/2); if(idx<0)idx=0; if(idx>=new_T)idx=new_T-1;
                memcpy(out+o,&feat_[(size_t)idx*NMEL],NMEL*sizeof(float)); o+=NMEL;
            }
        }
        float*shift=(float*)g("cmvn.shift")->data,*scale=(float*)g("cmvn.scale")->data;
        for(int i=0;i<nb;i++)for(int d=0;d<idim;d++)feats[(size_t)i*idim+d]=(feats[(size_t)i*idim+d]+shift[d])*scale[d];
        ggml_backend_t be=ggml_backend_cpu_init();
        ggml_init_params cp={(size_t)16*1024*1024,nullptr,true}; ggml_context*c=ggml_init(cp);
        ggml_tensor*x=ggml_new_tensor_2d(c,GGML_TYPE_F32,idim,nb); ggml_set_input(x);
        ggml_tensor*h=lin(c,g("encoder.in_linear1.linear.weight"),g("encoder.in_linear1.linear.bias"),x);
        h=lin(c,g("encoder.in_linear2.linear.weight"),g("encoder.in_linear2.linear.bias"),h); h=ggml_relu(c,h);
        for(int i=0;i<nl;i++){std::string p="encoder.fsmn."+std::to_string(i)+".";
            ggml_tensor*z=ggml_mul_mat(c,g(p+"linear.linear.weight"),h);
            ggml_tensor*fk=g(p+"fsmn_block.conv_left.weight");
            ggml_tensor*zp=ggml_pad_ext(c,z,0,0,lorder-1,0,0,0,0,0); ggml_tensor*accn=z;
            for(int j=0;j<lorder;j++){auto sl=ggml_view_2d(c,zp,pd,nb,zp->nb[1],(size_t)j*zp->nb[1]);auto wj=ggml_view_1d(c,fk,pd,(size_t)j*fk->nb[1]);accn=ggml_add(c,accn,ggml_mul(c,sl,wj));}
            ggml_tensor*a=lin(c,g(p+"affine.linear.weight"),g(p+"affine.linear.bias"),accn); h=ggml_relu(c,a);}
        h=lin(c,g("encoder.out_linear1.linear.weight"),g("encoder.out_linear1.linear.bias"),h);
        h=lin(c,g("encoder.out_linear2.linear.weight"),g("encoder.out_linear2.linear.bias"),h);
        h=ggml_soft_max(c,h); ggml_set_output(h);
        ggml_cgraph*gf=ggml_new_graph(c); ggml_build_forward_expand(gf,h);
        ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_cpu_buffer_type()); ggml_gallocr_alloc_graph(ga,gf);
        ggml_backend_tensor_set(x,feats.data(),0,ggml_nbytes(x)); ggml_backend_cpu_set_n_threads(be,nthreads);
        bool ok_=ggml_backend_graph_compute(be,gf)==GGML_STATUS_SUCCESS;
        if(ok_){ size_t old=sil_.size(); sil_.resize(old+(size_t)od*nb);
            ggml_backend_tensor_get(h,&sil_[old],0,ggml_nbytes(h)); }
        ggml_gallocr_free(ga); ggml_free(c); ggml_backend_free(be);
        return ok_;
    }

    // Drive the state machine over only newly-arrived frames; emit their events.
    std::vector<VadEvent> step_all(){
        std::vector<VadEvent> ev;
        int T=0;
        if(!forward_new_scores(T)){ return ev; }
        auto emit=[&](int s,int e){
            if(s<prev_end)s=prev_end; if(s<0)s=0; if(e>T)e=T;
            if(e>s){ ev.push_back({VAD_SEGMENT, s*FR, e*FR}); ev.push_back({VAD_SPEECH_STOP, e*FR, 0}); prev_end=e; }
        };
        for(int t=stepped; t<T; t++){
            if(t>0 && t%CHUNK==0){ if(st_sm==1||insp){acc+=60000; insp=1;} recompute(); }
            float sp=sil_[(size_t)t*od+0];
            int fs = ((1.0f-sp) >= sp + 0.5f) ? 1 : 0;
            wsum -= wbuf[wpos]; wsum += fs; wbuf[wpos]=fs; wpos=(wpos+1)%win;
            int ch;
            if(pre==0 && wsum>=s2s){pre=1; ch=3;}
            else if(pre==1 && wsum<=sp2s){pre=0; ch=1;}
            else ch = pre==0?0:2;
            if(ch==3){
                csil=0;
                if(st_sm==0){
                    cstart=t-start_lookback; if(cstart<prev_end)cstart=prev_end; if(cstart<0)cstart=0;
                    st_sm=1; ev.push_back({VAD_SPEECH_START, cstart*FR, 0});
                } else if(st_sm==1 && t-cstart+1>max_seg){ emit(cstart,t); reset_sm(); }
            } else if(ch==1||ch==2){
                csil=0;
                if(st_sm==1 && t-cstart+1>max_seg){ emit(cstart,t); reset_sm(); }
            } else {
                csil++;
                if(st_sm==1){
                    if(csil>=max_end_sil){ emit(cstart, t-end_lookback); reset_sm(); }
                    else if(soft_seg>0 && csil>=soft_sil && t-cstart+1>soft_seg){
                        // keep the same 100 ms speech tail the regular end-point keeps
                        emit(cstart, t-csil+lookahead_end); reset_sm();
                    }
                    else if(t-cstart+1>max_seg){ emit(cstart,t); reset_sm(); }
                }
            }
        }
        stepped = T;
        return ev;
    }

    bool in_speech() const { return st_sm==1; }
    // open utterance window (ms) relative to sample buffer start.
    void open_window(int& start_ms, int& end_ms) const {
        start_ms = st_sm==1 ? cstart*FR : 0;
        end_ms   = stepped*FR;
    }

    // End-of-stream flush: emit any open segment (mirrors the whole-file tail emit).
    std::vector<VadEvent> finalize(){
        std::vector<VadEvent> ev;
        int T=0;
        forward_new_scores(T);   // ensure scores exist for all current frames (no-op if already scored)
        if(T<1) return ev;
        if(st_sm==1){
            int s=cstart,e=T; if(s<prev_end)s=prev_end; if(s<0)s=0; if(e>T)e=T;
            if(e>s){ ev.push_back({VAD_SEGMENT, s*FR, e*FR}); ev.push_back({VAD_SPEECH_STOP, e*FR, 0}); }
            reset_sm();
        }
        return ev;
    }

    // Drop samples/frames before the just-finalized segment so later chunks only
    // re-process the current utterance. Safe while st_sm==0: the next cstart is
    // clamped to >= prev_end, which is >= the keep watermark. Returns the number
    // of 16k samples dropped so the caller can trim its session buffer in sync.
    int trim_after_segment(){
        if(st_sm==1) return 0;
        int keep=prev_end-start_lookback; if(keep<0)keep=0;
        keep=std::min(keep,(int)(sil_.size()/od));
        keep=std::min(keep,(int)(feat_.size()/NMEL));
        if(keep<=0) return 0;
        int ns=std::min(keep*SHIFT,(int)wav.size()); keep=ns/SHIFT;
        if(keep<=0) return 0;
        wav.erase(wav.begin(), wav.begin()+keep*SHIFT);
        feat_.erase(feat_.begin(), feat_.begin()+keep*NMEL);
        sil_.erase(sil_.begin(), sil_.begin()+keep*od);
        stepped-=keep; prev_end-=keep;
        if(cstart>0) cstart=std::max(0,cstart-keep);
        if(stepped<0)stepped=0; if(prev_end<0)prev_end=0;
        return keep*SHIFT;
    }

    // Full session reset (client input_audio_buffer.clear).
    void clear(){
        wav.clear(); feat_.clear(); sil_.clear();
        stepped=0; prev_end=0;
        reset_sm();
    }

    ~VadStream(){ if(ctx_w) ggml_free(ctx_w); }
};

// ===========================================================================
// Shared server context
// ===========================================================================
struct ServerCtx {
    SenseVoice sv;
    std::string vad_path;
    int nthreads=8, partial_ms=400, vad_maxseg=30000, vad_slot_ms=2000;
    int vad_soft_ms=10000, vad_soft_sil_ms=200;   // streaming soft split (see VadStream)
    int ws_idle_ms=60000;                         // 0 = no idle close
    size_t max_session_samples=(size_t)FS*300;
    bool keep_tags=false;
    std::atomic<int> connections{0};

    struct TranscriptResult {
        std::string text;
        std::vector<std::string> tags;
        std::string language;
        std::vector<TranscriptSegment> segs;
        double audio_ms = 0;
        bool vad_failed = false;
    };

    TranscriptResult transcribe_file(const std::vector<float>& wav){
        TranscriptResult out;
        out.audio_ms = (double)wav.size()/16.0;
        if(!vad_path.empty()){
            std::vector<std::pair<int,int>> s;
            if(!funasr_vad_segments(vad_path,wav,vad_maxseg,s)){
                fprintf(stderr,"[rest] VAD failed\n");
                out.vad_failed = true;
                return out;
            }
            for(auto&sg:s) out.segs.push_back({sg.first,sg.second,""});
        } else {
            if(!wav.empty()) out.segs.push_back({0,(int)(wav.size()/16),""});
        }
        for(auto&sg:out.segs){
            int off=(int)((int64_t)sg.start_ms*16), end=(int)((int64_t)sg.end_ms*16);
            if(end>(int)wav.size())end=(int)wav.size(); if(end-off<WINLEN)continue;
            std::vector<float> seg(wav.begin()+off,wav.begin()+end);
            auto r=sv.transcribe_wav(seg,keep_tags);
            sg.text=r.text;
            if(!out.text.empty() && !r.text.empty()) out.text+=" ";
            out.text+=r.text;
            for(auto&t:r.tags) out.tags.push_back(t);
            if(!r.language.empty() && out.language.empty()) out.language=r.language;
        }
        return out;
    }
};

// ===========================================================================
// WebSocket transcription session (OpenAI realtime transcription)
// ===========================================================================
class WsSession {
public:
    explicit WsSession(ServerCtx& ctx): ctx_(ctx){ last_rx_ms_ = now_ms(); }

    // Threading: the connection thread (this one) only reads frames and queues them, so
    // a close/EOF/reset from the client is seen immediately even while a decode is in
    // flight. A per-session worker thread does all VAD and model work. On disconnect the
    // worker is cancelled (queued audio dropped, an in-flight forward aborted through the
    // ggml abort callback) and joined, so the connection slot is released right away
    // instead of after the whole backlog has been transcribed for nobody.
    void run(httplib::ws::WebSocket& ws){
        send(ws, "{\"type\":\"session.created\",\"session\":{\"id\":\"sess_1\",\"object\":\"realtime.session\",\"model\":\"sensevoice\"}}");
        std::thread worker([this,&ws]{ work(ws); });
        std::string msg;
        for(;;){
            auto r = ws.read(msg);
            if(r==httplib::ws::ReadResult::Fail) break;   // close / error / idle timeout
            if(overflow_){
                ws.close(httplib::ws::CloseStatus::MessageTooBig, "session audio limit exceeded");
                break;
            }
            last_rx_ms_ = now_ms();
            bool binary = r==httplib::ws::ReadResult::Binary;
            std::string type = binary ? std::string() : json_field(msg,"type");
            {
                std::lock_guard<std::mutex> lk(q_mu_);
                if(type=="input_audio_buffer.commit") commits_queued_++;
                q_.push_back(InMsg{binary, std::move(type), std::move(msg)});
            }
            q_cv_.notify_one();
            msg.clear();
        }
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            closed_ = true;
            q_.clear();
        }
        q_cv_.notify_one();
        worker.join();
    }

private:
    struct InMsg { bool binary; std::string type; std::string data; };

    ServerCtx& ctx_;
    // reader -> worker queue
    std::mutex q_mu_;
    std::condition_variable q_cv_;
    std::deque<InMsg> q_;
    std::atomic<bool> closed_{false};      // client gone (or send failed): stop all work
    std::atomic<bool> overflow_{false};    // audio limit hit: reader closes on next frame
    std::atomic<long> commits_queued_{0};  // commits read so far
    std::atomic<long> commits_done_{0};    // commits handled by the worker
    std::atomic<long long> last_rx_ms_{0}; // arrival time of the last client message
    long long last_work_ms_ = 0;          // worker: end of the last batch it processed
    long long partial_gap_ms_ = 0;         // adaptive partial cadence (>= --partial-ms)

    // ggml abort callbacks: a final decode only stops when the client is gone; a partial
    // also yields to a pending commit so the end-of-turn transcript is never queued behind it.
    static bool abort_final(void* d){ return static_cast<WsSession*>(d)->closed_.load(); }
    static bool abort_partial(void* d){
        auto* s = static_cast<WsSession*>(d);
        return s->closed_.load() || s->commits_queued_.load()!=s->commits_done_.load();
    }

    void work(httplib::ws::WebSocket& ws){
        std::deque<InMsg> batch;
        while(!closed_){
            {
                std::unique_lock<std::mutex> lk(q_mu_);
                q_cv_.wait_for(lk, std::chrono::milliseconds(100), [&]{ return closed_.load() || !q_.empty(); });
                if(closed_) break;
                batch.swap(q_);
            }
            // Drain everything that arrived while the previous decode ran. The VAD is
            // still stepped once per received chunk, exactly as before (its scores
            // depend on chunk size), but no partial is decoded until the queue is empty.
            for(auto& m: batch){
                if(closed_ || overflow_) break;
                if(m.binary){
                    if(!append_raw_pcm16(m.data)){ on_overflow(ws); break; }
                    pump(ws, false);
                    continue;
                }
                if(m.type=="session.update")        on_session_update(m.data);
                else if(m.type=="input_audio_buffer.append"){
                    if(!on_append(m.data)){ on_overflow(ws); break; }
                    pump(ws, false);
                }
                else if(m.type=="input_audio_buffer.commit"){ commit(ws); commits_done_++; }
                else if(m.type=="input_audio_buffer.clear"){ clear(); }
                // response.create / response.cancel / conversation.* : accepted, ignored
            }
            if(!batch.empty()) last_work_ms_ = now_ms();
            batch.clear();
            if(closed_ || overflow_) break;
            if(ctx_.ws_idle_ms>0 &&
               now_ms()-std::max(last_rx_ms_.load(), last_work_ms_) >= ctx_.ws_idle_ms){
                // Client sent nothing for --ws-idle-timeout-s (counted from its last message
                // or from the end of the work it queued, whichever is later, so a slow final
                // decode never counts as client idleness): close with a proper close
                // frame (1001 "idle timeout"). The worker initiates it because the reader
                // is blocked in read(); the reader returns once the handshake completes.
                fprintf(stderr,"[ws] idle timeout, closing session\n");
                ws.close(httplib::ws::CloseStatus::GoingAway, "idle timeout");
                closed_ = true;
                break;
            }
            // Partials only run when no newer input is waiting, i.e. at most one partial
            // is ever in flight and it always covers the latest audio.
            bool idle;
            { std::lock_guard<std::mutex> lk(q_mu_); idle = q_.empty(); }
            pump(ws, idle);
        }
    }

    void on_overflow(httplib::ws::WebSocket& ws){
        send(ws, ev("error", "\"message\":\"session audio limit exceeded\""));
        clear();
        overflow_ = true;   // the reader thread closes the socket (it owns reads)
    }

    std::vector<float> samples_;         // session audio, 16k mono f32
    std::unique_ptr<VadStream> vad_;
    bool vad_on_ = true;
    std::string last_partial_;
    long long last_partial_ms_ = 0;
    size_t last_partial_sample_ = 0;    // samples_ size at last partial, for new-audio gating
    size_t quiet_samples_ = 0;          // audio (samples) received since the last audible chunk

    void send(httplib::ws::WebSocket& ws, const std::string& msg){
        if(closed_) return;
        if(!ws.send(msg)) closed_ = true;   // peer gone: stop decoding for nobody
    }
    std::string ev(const std::string& type, const std::string& extra){
        if(extra.empty()) return "{\"type\":"+json_str(type)+"}";
        return "{\"type\":"+json_str(type)+","+extra+"}";
    }

    void on_session_update(const std::string& p){
        auto sess = json_field(p,"session");
        if(sess.empty()) sess = p;
        auto td = json_field(sess,"turn_detection");
        auto td_type = json_field(td,"type");
        vad_on_ = !(td_type=="none" || td_type=="disabled" || td_type=="manual");
        if(ctx_.vad_path.empty()) vad_on_ = true;   // no VAD model -> treat input as one turn
        // input_audio_format pcm16 is assumed
    }

    // returns false when the session audio limit is exceeded
    bool on_append(const std::string& p){
        std::string b64 = json_field(p,"audio");
        if(b64.empty()) return true;
        std::string raw;
        if(!base64_decode(b64, raw) || raw.empty()){ fprintf(stderr,"[ws] bad base64\n"); return true; }
        return append_raw_pcm16(raw);
    }

    bool append_raw_pcm16(const std::string& raw){
        size_t n = raw.size()/2;
        if(n==0) return true;
        if(n>ctx_.max_session_samples || samples_.size()>ctx_.max_session_samples-n) return false;
        size_t off = samples_.size();
        samples_.resize(off+n);
        for(size_t i=0;i<n;i++){
            int16_t s; memcpy(&s, raw.data()+i*2, 2);
            samples_[off+i] = (float)s/32768.0f;
        }
        if(vad_ && vad_->ok) vad_->feed(samples_.data()+off, n);
        // Track the last chunk that carries actual signal so the idle-slot
        // cleanup below can retire a connection that stopped talking.
        double e=0;
        for(size_t i=0;i<n;i++) e += (double)samples_[off+i]*samples_[off+i];
        if(sqrt(e/(double)n) >= QUIET_RMS) quiet_samples_ = 0; else quiet_samples_ += n;
        return true;
    }

    void ensure_vad(){
        if(vad_) return;
        if(ctx_.vad_path.empty()) return;
        vad_.reset(new VadStream());
        vad_->nthreads = ctx_.nthreads;
        vad_->max_seg_ms = ctx_.vad_maxseg;
        vad_->soft_seg = ctx_.vad_soft_ms>0 ? ctx_.vad_soft_ms/vad_->FR : 0;
        vad_->soft_sil = std::max(1, ctx_.vad_soft_sil_ms/vad_->FR);
        if(!vad_->load(ctx_.vad_path.c_str())){ vad_.reset(); fprintf(stderr,"[ws] VAD load failed; single-turn mode\n"); return; }
        if(!samples_.empty()) vad_->feed(samples_.data(), samples_.size());
    }

    void feed_state(){
        if(!vad_on_ || ctx_.vad_path.empty()) return;
        ensure_vad();
    }

    // pump/commit run on the session worker thread.
    void pump(httplib::ws::WebSocket& ws, bool allow_partial){
        feed_state();
        if(vad_ && vad_on_){
            // Expected time slot expired: no audible chunk within vad_slot_ms.
            // Commit any still-open segment, then drop the VAD buffers and the
            // session audio so the connection parks until new chunks arrive.
            // Measured in received audio (quiet samples), plus wall clock only when
            // nothing at all has arrived: the worker's own decode time must not count,
            // or a slow decode turns ordinary pauses into a forced end-point and the
            // clear() below throws away audio that was not stepped through the VAD yet.
            long long t = now_ms();
            bool quiet = quiet_samples_ >= (size_t)ctx_.vad_slot_ms*16;
            bool silent_client = allow_partial && t - last_rx_ms_.load() >= ctx_.vad_slot_ms;
            if(quiet || silent_client){
                auto evs = vad_->finalize();
                for(auto&e: evs) if(e.type==VAD_SEGMENT) finalize_segment(ws, e.start_ms, e.end_ms);
                vad_->clear();
                samples_.clear();
                last_partial_.clear(); last_partial_ms_=0; last_partial_sample_=0;
                quiet_samples_ = 0;
            }
        }
        if(!vad_ || !vad_on_){ if(allow_partial) maybe_partial(ws); return; }
        auto evs = vad_->step_all();
        for(auto&e: evs){
            if(e.type==VAD_SPEECH_START){
                send(ws, ev("input_audio_buffer.speech_started", "\"audio_start_ms\":"+std::to_string(e.start_ms)));
            } else if(e.type==VAD_SPEECH_STOP){
                send(ws, ev("input_audio_buffer.speech_stopped",""));
            } else if(e.type==VAD_SEGMENT){
                finalize_segment(ws, e.start_ms, e.end_ms);
                trim_after_segment();
            }
        }
        if(allow_partial) maybe_partial(ws);
    }

    // Drop audio that is no longer reachable (everything before the last finalized
    // segment minus a lookback margin) from both the VAD and the session buffer.
    void trim_after_segment(){
        if(!vad_) return;
        int dropped = vad_->trim_after_segment();
        if(dropped>0){
            if(dropped>(int)samples_.size()) dropped=(int)samples_.size();
            samples_.erase(samples_.begin(), samples_.begin()+dropped);
            last_partial_sample_ = last_partial_sample_>dropped ? last_partial_sample_-dropped : 0;
        }
    }

    void maybe_partial(httplib::ws::WebSocket& ws){
        if(!vad_ || !vad_->in_speech()) return;
        long long t = now_ms();
        if(t - last_partial_ms_ < std::max<long long>(ctx_.partial_ms, partial_gap_ms_)) return;
        // Skip the full re-encode when essentially no new audio arrived since the
        // last partial (avoids burning CPU re-transcribing a static window).
        size_t new_audio = samples_.size()>last_partial_sample_ ? samples_.size()-last_partial_sample_ : 0;
        if(new_audio < 160*5) return;
        last_partial_ms_ = t;
        int s,e; vad_->open_window(s,e);
        if(e<=s) return;
        int off=(int)((int64_t)s*16), end=(int)((int64_t)e*16);
        if(end>(int)samples_.size()) end=(int)samples_.size();
        if(end-off<WINLEN) return;
        std::vector<float> seg(samples_.begin()+off, samples_.begin()+end);
        auto r = ctx_.sv.transcribe_wav(seg,false,abort_partial,this);
        // Keep partials to <= ~50% duty so long open segments cannot starve VAD and
        // final decodes (or a co-located service) on small CPUs.
        partial_gap_ms_ = 2*(now_ms()-t);
        if(r.aborted) return;
        std::string delta;
        if(r.text.size()>last_partial_.size() &&
           r.text.compare(0, last_partial_.size(), last_partial_)==0){
            delta = r.text.substr(last_partial_.size());
        } else if(!r.text.empty() && r.text!=last_partial_){
            delta = r.text;   // hypothesis revised: resend full text
        }
        last_partial_ = r.text;
        last_partial_sample_ = samples_.size();
        if(delta.empty()) return;
        std::string item_id = "it_"+std::to_string(s);
        send(ws, ev("conversation.item.input_audio_transcription.delta",
            "\"item_id\":"+json_str(item_id)+",\"delta\":"+json_str(delta)));
    }

    void finalize_segment(httplib::ws::WebSocket& ws, int sms, int ems){
        int off=(int)((int64_t)sms*16), end=(int)((int64_t)ems*16);
        if(end>(int)samples_.size())end=(int)samples_.size();
        if(end-off<WINLEN) return;
        std::vector<float> seg(samples_.begin()+off, samples_.begin()+end);
        auto r = ctx_.sv.transcribe_wav(seg,false,abort_final,this);
        if(r.aborted) return;
        last_partial_.clear();
        std::string item_id = "it_"+std::to_string(sms);
        send(ws, ev("input_audio_buffer.committed",""));
        send(ws, ev("conversation.item.created",
            "\"item\":{\"id\":"+json_str(item_id)+",\"object\":\"realtime.item\",\"type\":\"input_audio\"}"));
        std::string msg = "{\"type\":\"conversation.item.input_audio_transcription.completed\","
                          "\"item_id\":"+json_str(item_id)+",\"transcript\":"+json_str(r.text);
        if(!r.language.empty()) msg += ",\"language\":"+json_str(r.language);
        msg += "}";
        send(ws, msg);
    }

    void commit(httplib::ws::WebSocket& ws){
        if(!vad_ || !vad_on_){
            // no VAD / manual mode: transcribe everything accumulated
            if((int)samples_.size()>=WINLEN){
                finalize_segment(ws, 0, (int)(samples_.size()/16));
                samples_.clear(); last_partial_.clear(); last_partial_sample_=0;
            }
            return;
        }
        auto evs = vad_->finalize();
        for(auto&e: evs) if(e.type==VAD_SEGMENT) finalize_segment(ws, e.start_ms, e.end_ms);
        trim_after_segment();
    }

    void clear(){
        samples_.clear(); last_partial_.clear(); last_partial_ms_=0; last_partial_sample_=0;
        quiet_samples_ = 0;
        if(vad_) vad_->clear();
    }
};

static void usage(const char* a0){
    fprintf(stderr,
"\n"
"sensevoice-server - OpenAI-compatible STT server (SenseVoiceSmall on ggml)\n"
"\n"
"Usage:\n"
"  %s -m <sensevoice.gguf> [-vad <fsmn-vad.gguf>] [host [port]]\n"
"\n"
"Options:\n"
"  -m, --model <file>     SenseVoice GGUF (required)\n"
"  -vad, --vad <file>     FSMN-VAD GGUF; used for sentence/utterance end-pointing\n"
"  -ngl, --ngl <N>        layers offloaded to CUDA GPU; any value > 0 runs the whole\n"
"                         model on device 0 (default 0 = CPU). Needs a GGML_CUDA build.\n"
"      --keep-tags        keep <|lang|>/<|emo|>/<|event|> meta tokens in text\n"
"      --threads <N>      ggml compute threads (default 8)\n"
      "      --partial-ms <N>   partial-transcription cadence for WS streaming (default 400)\n"
      "      --vad-maxseg <ms>  max single VAD segment, ms (default 30000)\n"
      "      --vad-slot-ms <ms>  idle slot: when no audible chunk arrives within this window,\n"
      "                         the VAD buffers are finalized and reset until new chunks come\n"
      "                         in (default 2000; idle sessions stop burning CPU)\n"
"      --read-timeout-s <N> socket read timeout, s (default 300)\n"
"      --web <dir>        serve a static directory at / (mic webui; open http://host:port/)\n"
"      --max-connections <N> maximum active WebSocket clients (default 4)\n"
"      --max-audio-seconds <N> maximum decoded REST audio or buffered WS audio (default 300)\n"
"      --ws-idle-timeout-s <N> close a realtime session (close frame 1001 \"idle timeout\")\n"
"                         after N s without a client message; frees the slot of clients that\n"
"                         stopped sending or vanished without a FIN (default 60; 0 = off)\n"
"      --vad-soft-split-ms <ms> streaming: once an open segment is longer than this, end it\n"
"                         at the next short pause (default 10000; 0 = off)\n"
"      --vad-soft-sil-ms <ms> pause length for the soft split (default 200)\n"
"  -h, --help             show this help\n"
"\n"
"Positional:\n"
"  host                   bind address (default 127.0.0.1)\n"
"  port                   listen port   (default 8040)\n"
"\n"
"Endpoints:\n"
"  POST /v1/audio/transcriptions        OpenAI file transcription (multipart form-data)\n"
"  GET  /v1/models / GET /health\n"
"  WS   /v1/realtime?intent=transcription   streaming transcription (VAD turn detection)\n"
"\n",
    a0);
}

int main(int argc, char** argv){
    std::string model_path, vad_path, webdir, host="127.0.0.1";
    int port=8040, threads=8, partial_ms=400, vad_maxseg=30000, vad_slot_ms=2000, read_timeout_s=300;
    int max_connections=4, max_audio_seconds=300, ws_idle_timeout_s=60;
    int vad_soft_ms=10000, vad_soft_sil_ms=200;
    int ngl=0;
    bool keep_tags=false;
    std::vector<std::string> pos;
    int i=1;
    for(;i<argc;i++){
        std::string a=argv[i];
        auto take=[&](const char* f)->const char*{ if(i+1<argc) return argv[++i]; fprintf(stderr,"missing value for %s\n",f); exit(1); return nullptr; };
        if(a=="-h"||a=="--help"){ usage(argv[0]); return 0; }
        else if(a=="-m"||a=="--model") model_path=take(a.c_str());
        else if(a=="-vad"||a=="--vad") vad_path=take(a.c_str());
        else if(a=="-ngl"||a=="--ngl") ngl=atoi(take("-ngl"));
        else if(a=="--web") webdir=take(a.c_str());
        else if(a=="--threads") threads=atoi(take("--threads"));
        else if(a=="--partial-ms") partial_ms=atoi(take("--partial-ms"));
        else if(a=="--vad-maxseg") vad_maxseg=atoi(take("--vad-maxseg"));
        else if(a=="--vad-slot-ms") vad_slot_ms=atoi(take("--vad-slot-ms"));
        else if(a=="--read-timeout-s") read_timeout_s=atoi(take("--read-timeout-s"));
        else if(a=="--max-connections"||a=="--max-conn") max_connections=atoi(take(a.c_str()));
        else if(a=="--max-audio-seconds") max_audio_seconds=atoi(take("--max-audio-seconds"));
        else if(a=="--ws-idle-timeout-s") ws_idle_timeout_s=atoi(take("--ws-idle-timeout-s"));
        else if(a=="--vad-soft-split-ms") vad_soft_ms=atoi(take("--vad-soft-split-ms"));
        else if(a=="--vad-soft-sil-ms") vad_soft_sil_ms=atoi(take("--vad-soft-sil-ms"));
        else if(a=="--keep-tags") keep_tags=true;
        else if(!a.empty() && a[0]=='-'){ fprintf(stderr,"unknown option %s\n",a.c_str()); return 1; }
        else pos.push_back(a);
    }
    if(pos.size()>0) host=pos[0];
    if(pos.size()>1) port=atoi(pos[1].c_str());
    if(threads<1) threads=1;
    if(partial_ms<50) partial_ms=50;
    if(vad_slot_ms<50) vad_slot_ms=50;
    if(max_connections<1 || max_connections>128){ fprintf(stderr,"--max-connections must be between 1 and 128\n"); return 1; }
    if(max_audio_seconds<1 || max_audio_seconds>3600){ fprintf(stderr,"--max-audio-seconds must be between 1 and 3600\n"); return 1; }
    if(model_path.empty()){ usage(argv[0]); return 1; }

    ServerCtx ctx;
    ctx.nthreads=threads; ctx.partial_ms=partial_ms; ctx.vad_maxseg=vad_maxseg;
    ctx.vad_slot_ms=vad_slot_ms;
    ctx.vad_soft_ms=vad_soft_ms<0?0:vad_soft_ms; ctx.vad_soft_sil_ms=vad_soft_sil_ms<10?10:vad_soft_sil_ms;
    ctx.ws_idle_ms=ws_idle_timeout_s>0 ? ws_idle_timeout_s*1000 : 0;
    ctx.max_session_samples=(size_t)FS*(size_t)max_audio_seconds;
    ctx.keep_tags=keep_tags; ctx.vad_path=vad_path;

    fprintf(stderr,"[server] loading %s ...\n",model_path.c_str());
    if(!ctx.sv.load(model_path.c_str())){ fprintf(stderr,"load model failed\n"); return 1; }
    ctx.sv.nthreads=threads;
    if(!ctx.sv.init_gpu(ngl)){ fprintf(stderr,"GPU offload failed\n"); return 1; }
    fprintf(stderr,"[server] model loaded (d_model=%d layers=%d tp=%d vocab=%d ngl=%d%s)\n",
            ctx.sv.m.c.d_model, ctx.sv.m.c.num_blocks, ctx.sv.m.c.tp_blocks, ctx.sv.m.c.vocab,
            ctx.sv.ngl,
            vad_path.empty()?"":", vad enabled");

    httplib::Server svr;
    // Keep two HTTP workers available while long-lived WebSockets occupy the
    // dynamically sized remainder. The queue and thread count are bounded.
    svr.new_task_queue = [max_connections](){
        return static_cast<httplib::TaskQueue*>(new httplib::ThreadPool(
            2, (size_t)max_connections+2, (size_t)max_connections+16));
    };
    svr.set_payload_max_length((size_t)64*1024*1024);
    svr.set_read_timeout(read_timeout_s, 0);
    svr.set_write_timeout(read_timeout_s, 0);
    svr.set_keep_alive_max_count(8);
    // The session worker sends the 1001 "idle timeout" close frame at N s; the socket read
    // timeout (N+2 s, no frame at all incl. pongs) then ends the connection even if the peer
    // never completes the close handshake or vanished without a FIN.
    // Pings (answered by any live client's WebSocket stack) keep the read timeout from
    // firing on a quiet but alive client, e.g. one waiting for a long final decode.
    g_ws_idle_timeout_s = ws_idle_timeout_s>0 ? ws_idle_timeout_s+2 : 300;
    if(ws_idle_timeout_s>0)
        svr.set_websocket_ping_interval(std::min<time_t>(30, std::max<time_t>(1, ws_idle_timeout_s/3)));

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res){
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr.Get("/v1/models", [](const httplib::Request&, httplib::Response& res){
        res.set_content("{\"object\":\"list\",\"data\":[{\"id\":\"sensevoice-small\",\"object\":\"model\",\"created\":0,\"owned_by\":\"funasr\"}]}",
                        "application/json");
    });

    svr.Post("/v1/audio/transcriptions", [&](const httplib::Request& req, httplib::Response& res){
        if(!req.form.has_file("file")){
            res.status=400;
            res.set_content("{\"error\":{\"message\":\"missing 'file' field\",\"type\":\"invalid_request_error\"}}","application/json");
            return;
        }
        auto f = req.form.get_file("file");
        std::string fmt = req.form.has_field("response_format") ? req.form.get_field("response_format") : "json";
        bool stream = req.form.has_field("stream") && req.form.get_field("stream")=="true";
        if(fmt!="json" && fmt!="text" && fmt!="verbose_json" && fmt!="srt" && fmt!="vtt"){
            res.status=400;
            res.set_content("{\"error\":{\"message\":\"unsupported response_format\",\"type\":\"invalid_request_error\"}}","application/json");
            return;
        }
        std::vector<float> wav;
        auto decode_result=decode_audio_mem(f.content.data(), f.content.size(), ctx.max_session_samples, wav);
        if(decode_result==AudioDecodeResult::TooLarge){
            res.status=413;
            res.set_content("{\"error\":{\"message\":\"audio duration exceeds configured limit\",\"type\":\"invalid_request_error\"}}","application/json");
            return;
        }
        if(decode_result!=AudioDecodeResult::Ok){
            res.status=400;
            res.set_content("{\"error\":{\"message\":\"cannot decode audio\",\"type\":\"invalid_request_error\"}}","application/json");
            return;
        }
        fprintf(stderr,"[rest] transcribe %s (%.1fs, %s)\n", f.filename.c_str(), (double)wav.size()/16000.0, fmt.c_str());
        auto tr = ctx.transcribe_file(wav);
        if(tr.vad_failed){
            res.status=500;
            res.set_content("{\"error\":{\"message\":\"VAD inference failed\",\"type\":\"server_error\"}}","application/json");
            return;
        }
        if(stream){
            // SSE: one transcript.text.delta per recognized segment, then the full text.
            std::vector<std::string> evs;
            for(const auto& segment : tr.segs){
                if(!segment.text.empty()) evs.push_back("event: transcript.text.delta\ndata: {\"type\":\"transcript.text.delta\",\"delta\":"+json_str(segment.text)+"}\n\n");
            }
            evs.push_back("event: transcript.text.done\ndata: {\"type\":\"transcript.text.done\",\"text\":"+json_str(tr.text)+"}\n\n");
            size_t idx=0;
            res.set_chunked_content_provider("text/event-stream", [evs,idx](size_t, httplib::DataSink& sink) mutable -> bool {
                if(idx>=evs.size()){ return true; }
                if(!sink.write(evs[idx].c_str(), evs[idx].size())) return false;
                idx++;
                if(idx>=evs.size()){ sink.done(); }
                return true;
            }, nullptr);
            return;
        }
        std::string body; std::string ct="application/json";
        if(fmt=="text"){ body=tr.text; ct="text/plain"; }
        else if(fmt=="verbose_json"){
            std::string segs=format_verbose_segments(tr.segs);
            body = "{\"task\":\"transcribe\",\"language\":"+json_str(tr.language)+",\"duration\":"+
                   std::to_string(tr.audio_ms/1000.0)+",\"text\":"+json_str(tr.text)+",\"segments\":["+segs+"]}";
            ct="application/json";
        }
        else if(fmt=="srt" || fmt=="vtt"){
            body=format_subtitles(tr.segs,fmt=="vtt");
            ct = fmt=="srt"?"application/x-subrip":"text/vtt";
        }
        else { body = "{\"text\":"+json_str(tr.text)+"}"; }
        res.set_content(body, ct);
    });

    svr.WebSocket("/v1/realtime", [&](const httplib::Request&, httplib::ws::WebSocket& ws){
        int active=ctx.connections.fetch_add(1)+1;
        if(active>max_connections){
            ctx.connections--;
            ws.send("{\"type\":\"error\",\"message\":\"connection limit exceeded\"}");
            ws.close(httplib::ws::CloseStatus::PolicyViolation, "connection limit exceeded");
            return;
        }
        struct ConnectionGuard {
            std::atomic<int>& count;
            ~ConnectionGuard(){ count--; }
        } guard{ctx.connections};
        fprintf(stderr,"[ws] client connected (%d/%d active)\n", active, max_connections);
        WsSession sess(ctx);
        sess.run(ws);
        fprintf(stderr,"[ws] client disconnected\n");
    });

    fprintf(stderr,"[server] listening on %s:%d\n", host.c_str(), port);
    if(!webdir.empty()){
        if(!svr.set_mount_point("/", webdir))
            fprintf(stderr,"[server] warning: cannot serve %s (no index.html?)\n", webdir.c_str());
        else
            fprintf(stderr,"[server] serving webui from %s at /\n", webdir.c_str());
    }
    if(!svr.bind_to_port(host, port)){
        fprintf(stderr,"[server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    fprintf(stderr,"[server]   POST /v1/audio/transcriptions\n");
    fprintf(stderr,"[server]   WS   /v1/realtime?intent=transcription\n");
    svr.listen_after_bind();
    return 0;
}
