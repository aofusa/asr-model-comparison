#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static char * amcp_dup_cstr(const std::string & value) {
    char * out = static_cast<char *>(std::malloc(value.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

static void amcp_set_error(char ** out_error, const std::string & message) {
    if (out_error) {
        *out_error = amcp_dup_cstr(message);
    }
}

static std::string amcp_detokenize(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    if (tokens.empty()) {
        return {};
    }

    int32_t size = llama_detokenize(
        vocab,
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        nullptr,
        0,
        true,
        false);
    if (size < 0) {
        size = -size;
    }
    if (size <= 0) {
        return {};
    }

    std::string text(static_cast<size_t>(size), '\0');
    const int32_t written = llama_detokenize(
        vocab,
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        text.data(),
        size,
        true,
        false);
    if (written < 0) {
        return {};
    }
    text.resize(static_cast<size_t>(written));
    return text;
}

extern "C" int32_t amcp_voxtral_realtime_transcribe(
    const char * model_path,
    const char * mmproj_path,
    const float * pcm_samples,
    size_t n_samples,
    uint32_t context_size,
    uint32_t batch_size,
    uint32_t max_tokens,
    int32_t n_gpu_layers,
    bool use_gpu,
    bool print_timings,
    char ** out_text,
    char ** out_error) {
    if (out_text) {
        *out_text = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }
    if (!model_path || !mmproj_path || !pcm_samples || n_samples == 0 || !out_text) {
        amcp_set_error(out_error, "invalid Voxtral Realtime bridge arguments");
        return -1;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        amcp_set_error(out_error, std::string("failed to load Voxtral Realtime GGUF model: ") + model_path);
        return -2;
    }

    const uint32_t n_batch = std::max<uint32_t>(1, std::min(batch_size, context_size));
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = context_size;
    context_params.n_batch = n_batch;
    context_params.n_ubatch = n_batch;

    llama_context * lctx = llama_init_from_model(model, context_params);
    if (!lctx) {
        llama_model_free(model);
        amcp_set_error(out_error, "failed to create Voxtral Realtime llama context");
        return -3;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = use_gpu;
    mtmd_params.print_timings = print_timings;

    mtmd_context * mctx = mtmd_init_from_file(mmproj_path, model, mtmd_params);
    if (!mctx) {
        llama_free(lctx);
        llama_model_free(model);
        amcp_set_error(out_error, std::string("failed to load Voxtral Realtime mmproj: ") + mmproj_path);
        return -4;
    }

    if (!mtmd_support_voxtral_realtime(mctx)) {
        mtmd_free(mctx);
        llama_free(lctx);
        llama_model_free(model);
        amcp_set_error(out_error, "mmproj does not report Voxtral Realtime support");
        return -5;
    }

    llama_pos n_past = 0;
    std::vector<llama_token> output_tokens;
    const int32_t result = mtmd_helper_eval_voxtral_realtime(
        mctx,
        lctx,
        model_path,
        pcm_samples,
        n_samples,
        static_cast<int32_t>(n_batch),
        static_cast<int32_t>(max_tokens),
        &n_past,
        &output_tokens);

    if (result != 0) {
        mtmd_free(mctx);
        llama_free(lctx);
        llama_model_free(model);
        amcp_set_error(out_error, "Voxtral Realtime dual-stream evaluation failed");
        return result;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    *out_text = amcp_dup_cstr(amcp_detokenize(vocab, output_tokens));

    mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);
    return *out_text ? 0 : -6;
}

extern "C" void amcp_voxtral_realtime_free(char * value) {
    std::free(value);
}
