#include <fstream>
#include <iostream>

#include "common.h"
#include "llama.h"


int run_llama_embedding(llama_context * ctx, gpt_params params, std::ostream *outfile) {
    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    int n_past = 0;

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // tokenize the prompt
    auto embd_inp = ::llama_tokenize(ctx, params.prompt, true);

    // determine newline token
    auto llama_token_newline = ::llama_tokenize(ctx, "\n", false);

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
        }
        fprintf(stderr, "\n");
    }

    if (params.embedding){
        if (embd_inp.size() > 0) {
            if (llama_eval(ctx, embd_inp.data(), embd_inp.size(), n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
        }

        const int n_embd = llama_n_embd(ctx);
        const auto embeddings = llama_get_embeddings(ctx);

        for (int i = 0; i < n_embd; i++) {
            (*outfile) << embeddings[i] << " ";
        }
        (*outfile) << "\n";
    }

    llama_print_timings(ctx);
    return 0;
}