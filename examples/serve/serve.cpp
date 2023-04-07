#include "common.h"
#include "llama.h"

#include "crow.h"
#include "../main/run_llama.cpp"

auto const BINDPORT = 8081;

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/llama-7B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false)
        return 1;

    if (params.n_ctx > 2048)
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);

    if (params.seed <= 0)
        params.seed = time(NULL);

    llama_context * ctx;
    
    // load the model
    {
        auto lparams = llama_context_default_params();

        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.use_mlock  = params.use_mlock;
        
        // ctx = llama_init_from_file(params.model.c_str(), lparams);

        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    crow::SimpleApp app;

    /// Python server will send a file name to you.
    /// You should open that file and give the pointer to run_llama.
    /// run_llama will keep writing the output to it.
    /// The python server will keep reading from that file just like it reads
    /// from the stdout of the main process.
    ///
    /// We are doing this because this is probably the simplest way
    /// to get streaming to work here. Otherwise I'll have to handle sockets.

    CROW_ROUTE(app, "/completion").methods("POST"_method)
    ([](const crow::request& req){
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(crow::status::BAD_REQUEST);
        
        // Open the tempfile and get a FP.
        // Set run params from body
        // run_llama(ctx, params, tempfile);

        return crow::response(crow::status::OK);
    });

    CROW_ROUTE(app, "/embedding").methods("POST"_method)
    ([](const crow::request& req){
        auto body = crow::json::load(req.body);
        if (!body) return crow::response(crow::status::BAD_REQUEST);
        
        // Open the tempfile and get a FP.
        // Set run params from body
        // run_llama(ctx, params, tempfile);

        return crow::response(crow::status::OK);
    });

    app.port(BINDPORT).multithreaded().run();
    return 0;
}