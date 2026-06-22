#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstdint>
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

static int32_t amcp_eval_tokens(llama_context * lctx, const std::vector<llama_token> & tokens, int32_t n_batch, llama_pos * n_past) {
    for (size_t offset = 0; offset < tokens.size(); offset += static_cast<size_t>(n_batch)) {
        const int32_t n_tokens = static_cast<int32_t>(std::min<size_t>(static_cast<size_t>(n_batch), tokens.size() - offset));
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        batch.n_tokens = n_tokens;
        for (int32_t i = 0; i < n_tokens; ++i) {
            batch.token[i] = tokens[offset + static_cast<size_t>(i)];
            batch.pos[i] = *n_past + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tokens - 1) ? 1 : 0;
        }
        const int32_t result = llama_decode(lctx, batch);
        llama_batch_free(batch);
        if (result != 0) {
            return result;
        }
        *n_past += n_tokens;
    }
    return 0;
}

static bool amcp_tokenize_prompt(const llama_vocab * vocab, const char * prompt, std::vector<llama_token> & tokens) {
    const int32_t prompt_len = static_cast<int32_t>(std::strlen(prompt));
    int32_t n_tokens = llama_tokenize(vocab, prompt, prompt_len, nullptr, 0, true, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }
    if (n_tokens <= 0) {
        return false;
    }
    tokens.resize(static_cast<size_t>(n_tokens));
    const int32_t written = llama_tokenize(vocab, prompt, prompt_len, tokens.data(), n_tokens, true, true);
    if (written < 0) {
        return false;
    }
    tokens.resize(static_cast<size_t>(written));
    return !tokens.empty();
}

static std::string amcp_apply_translation_chat_template(llama_model * model, const char * prompt) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        return prompt;
    }

    const llama_chat_message messages[] = {
        {
            "system",
            "You are a precise translation engine. Return only the translated text, with no explanation.",
        },
        { "user", prompt },
    };

    int32_t size = llama_chat_apply_template(tmpl, messages, 2, true, nullptr, 0);
    if (size <= 0) {
        return prompt;
    }

    std::string formatted(static_cast<size_t>(size), '\0');
    const int32_t written = llama_chat_apply_template(
        tmpl,
        messages,
        2,
        true,
        formatted.data(),
        size);
    if (written <= 0) {
        return prompt;
    }
    formatted.resize(static_cast<size_t>(written));
    return formatted;
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

extern "C" int32_t amcp_voxtral_realtime_generate_text(
    const char * model_path,
    const char * prompt,
    uint32_t context_size,
    uint32_t batch_size,
    uint32_t max_tokens,
    int32_t n_gpu_layers,
    char ** out_text,
    char ** out_error) {
    if (out_text) {
        *out_text = nullptr;
    }
    if (out_error) {
        *out_error = nullptr;
    }
    if (!model_path || !prompt || !out_text) {
        amcp_set_error(out_error, "invalid Voxtral Realtime text generation arguments");
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
        amcp_set_error(out_error, "failed to create Voxtral Realtime text generation context");
        return -3;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string formatted_prompt = amcp_apply_translation_chat_template(model, prompt);
    std::vector<llama_token> prompt_tokens;
    if (!amcp_tokenize_prompt(vocab, formatted_prompt.c_str(), prompt_tokens)) {
        llama_free(lctx);
        llama_model_free(model);
        amcp_set_error(out_error, "failed to tokenize Voxtral Realtime text prompt");
        return -4;
    }

    llama_pos n_past = 0;
    int32_t result = amcp_eval_tokens(lctx, prompt_tokens, static_cast<int32_t>(n_batch), &n_past);
    if (result != 0) {
        llama_free(lctx);
        llama_model_free(model);
        amcp_set_error(out_error, "Voxtral Realtime text prompt evaluation failed");
        return result;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    if (!sampler) {
        llama_free(lctx);
        llama_model_free(model);
        amcp_set_error(out_error, "failed to create Voxtral Realtime text sampler");
        return -5;
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.15f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.90f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.30f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));
    for (const llama_token token : prompt_tokens) {
        llama_sampler_accept(sampler, token);
    }

    std::vector<llama_token> output_tokens;
    output_tokens.reserve(max_tokens);
    for (uint32_t i = 0; i < max_tokens; ++i) {
        const llama_token token = llama_sampler_sample(sampler, lctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        llama_sampler_accept(sampler, token);
        output_tokens.push_back(token);
        std::vector<llama_token> next = { token };
        result = amcp_eval_tokens(lctx, next, 1, &n_past);
        if (result != 0) {
            break;
        }
    }

    *out_text = amcp_dup_cstr(amcp_detokenize(vocab, output_tokens));
    llama_sampler_free(sampler);
    llama_free(lctx);
    llama_model_free(model);
    return *out_text ? 0 : -5;
}
