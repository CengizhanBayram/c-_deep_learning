#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <memory>

// ONNX Runtime başlık dosyaları
#include <onnxruntime_cxx_api.h>

// Faiss başlık dosyaları
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

// llama.cpp başlık dosyaları (projenize göre yolu düzenleyin)
#include "llama.h"

// Tokenizer için basit bir kütüphane (örneğin, sentencepiece veya kendi uygulamanız)
// Bu örnekte, basitlik adına tokenizer kısmı atlanmış ve doğrudan ONNX modeline
// uygun formatta girdi sağlanacağı varsayılmıştır. Gerçek bir uygulamada
// Hugging Face tokenizer'larının C++ portlarından biri kullanılabilir.

// Vektörleri yazdırmak için yardımcı fonksiyon
void print_vector(const std::vector<float>& vec) {
    for (float val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// Metin Gömme Sınıfı
class TextEmbedder {
public:
    TextEmbedder(const char* model_path) {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "RAG_Example");
        session = Ort::Session(env, model_path, Ort::SessionOptions{nullptr});
    }

    std::vector<float> get_embedding(const std::vector<int64_t>& input_ids) {
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info,
                                                                  const_cast<int64_t*>(input_ids.data()),
                                                                  input_ids.size(),
                                                                  input_shape.data(),
                                                                  input_shape.size());

        const char* input_names[] = {"input_ids"};
        const char* output_names[] = {"sentence_embedding"};

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        float* floatarr = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        return std::vector<float>(floatarr, floatarr + output_size);
    }

private:
    Ort::Env env;
    Ort::Session session{nullptr};
};

int main() {
    // --- 1. Veri Hazırlığı ve Gömme (Embedding) ---

    // Bilgi tabanımızdaki metinler
    std::vector<std::string> documents = {
        "C++ is a general-purpose programming language created by Bjarne Stroustrup.",
        "Faiss is a library for efficient similarity search and clustering of dense vectors.",
        "ONNX Runtime is a cross-platform inferencing and training accelerator.",
        "llama.cpp is a port of Facebook's LLaMA model in C/C++."
    };

    // ONNX modelini ve tokenizer'ı yükle
    TextEmbedder embedder("path/to/your/all-MiniLM-L6-v2.onnx");

    // Gerçek bir uygulamada, burada bir tokenizer kullanarak metinleri
    // `input_ids`'ye dönüştürmeniz gerekir. Bu örnekte temsili `input_ids` kullanılmıştır.
    std::vector<std::vector<int64_t>> document_input_ids = {
        {101, 2182, 2182, 2003, 1037, ...}, // Temsili token ID'leri
        {101, 2524, 2319, 2003, 1037, ...},
        {101, 2524, 2319, 2003, 1037, ...},
        {101, 2524, 2319, 2003, 1037, ...}
    };

    std::vector<std::vector<float>> document_embeddings;
    for (const auto& ids : document_input_ids) {
        document_embeddings.push_back(embedder.get_embedding(ids));
    }

    int embedding_dim = document_embeddings[0].size();

    // --- 2. Faiss ile Vektör İndeksi Oluşturma ---

    faiss::IndexFlatL2 index(embedding_dim);
    for (const auto& emb : document_embeddings) {
        index.add(1, emb.data());
    }

    std::cout << "Faiss index created with " << index.ntotal << " vectors." << std::endl;

    // --- 3. Arama (Retrieval) ---

    std::string user_query = "What is Faiss?";
    // Yine, sorguyu token'lara ayırıp embedding'ini almamız gerekiyor.
    std::vector<int64_t> query_input_ids = {101, 2054, 2003, 2524, 2319, 102}; // Temsili
    std::vector<float> query_embedding = embedder.get_embedding(query_input_ids);

    int k = 2; // En benzer 2 dokümanı getir
    std::vector<long> retrieved_indices(k);
    std::vector<float> retrieved_distances(k);

    index.search(1, query_embedding.data(), k, retrieved_distances.data(), retrieved_indices.data());

    std::string context = "";
    for (long idx : retrieved_indices) {
        context += documents[idx] + "\n";
    }

    std::cout << "\nRetrieved Context:\n" << context << std::endl;

    // --- 4. Yanıt Üretme (Generation) ---

    // llama.cpp modelini yükle
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file("path/to/your/model.gguf", model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: failed to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    // LLM için prompt oluşturma
    std::string prompt = "Context:\n" + context + "\nQuestion: " + user_query + "\nAnswer:";

    // Prompt'u token'lara ayır
    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx, prompt.c_str(), true);

    const int n_len = 256; // Üretilecek maksimum token sayısı

    // LLM çıkarımını çalıştır
    llama_eval(ctx, tokens_list.data(), tokens_list.size(), 0, 8);

    std::cout << "\nGenerated Answer:\n";
    for (int i = 0; i < n_len; i++) {
        auto token_id = llama_sample_token_greedy(ctx, NULL);
        if (token_id == llama_token_eos(ctx)) {
            break;
        }
        std::cout << llama_token_to_piece(ctx, token_id);
        fflush(stdout);
        llama_eval(ctx, &token_id, 1, llama_get_kv_cache_token_count(ctx), 8);
    }
    std::cout << std::endl;

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return 0;
}
