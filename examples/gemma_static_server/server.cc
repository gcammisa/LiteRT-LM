/*
 * Gemma-3n OpenAI-compatible chat server (CPU-only, fully static)
 * --------------------------------------------------------------
 * Build deps (same as before):
 *
 *   "//runtime/core:engine_impl",
 *   "//runtime/engine:engine_interface",
 *   "//runtime/engine:engine_settings",
 *   "//runtime/engine:io_types",
 *   "@com_github_yhirose_cpp_httplib//:cpp_httplib",
 *   "@com_github_nlohmann_json//:json",
 *   "@com_github_jarro2783_cxxopts//:cxxopts"
 *
 * plus copts = ["-std=c++20"], linkopts = ["-static"],
 * linkstatic = True, features = ["fully_static_link"].
 */

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <cxxopts.hpp>
#include "runtime/components/sampling_cpu_util.h"

#include <memory>
#include <string>

#include "runtime/engine/engine.h"
#include "runtime/engine/io_types.h"

using json = nlohmann::json;
using namespace litert::lm;

/* ---------- singleton keeps the model in RAM ---------- */
class EngineHolder {
 public:
  static Engine& get(const std::string& model_path) {
    static EngineHolder self(model_path);
    return *self.engine_;
  }
 private:
  explicit EngineHolder(const std::string& path) {
    auto assets   = ModelAssets::Create(path).value();
    auto settings = EngineSettings::CreateDefault(assets, Backend::CPU).value();
    engine_       = Engine::CreateEngine(settings).value();
  }
  std::unique_ptr<Engine> engine_;
};

/* ------------- helpers ------------- */
json mk_err(int code, std::string_view type, std::string_view msg) {
  return {{"error", {{"code", code}, {"type", type}, {"message", msg}}}};
}

std::string generate(const Engine& eng,
                     const std::string& prompt,
                     int max_tokens) {
  SessionConfig cfg = SessionConfig::CreateDefault();          // no tunables in v0.6.1
  auto sess  = eng.CreateSession(cfg).value();
  auto resp  = sess->GenerateContent({InputText(prompt)}).value();
  std::string out = std::string(resp.GetResponseTextAt(0).value());
  if (max_tokens > 0 && static_cast<int>(out.size()) > max_tokens)
    out.resize(max_tokens);
  return out;
}

/* ------------------------ main ------------------------ */
int main(int argc, char** argv) {
  /* CLI flags */
  cxxopts::Options cli("gemma_static_server", "OpenAI-compatible Gemma-3n API");
  cli.add_options()
      ("model", "Path to .litertlm model",  cxxopts::value<std::string>())
      ("host",  "Listen address",           cxxopts::value<std::string>()->default_value("0.0.0.0"))
      ("port",  "Listen port",              cxxopts::value<int>()->default_value("8000"))   // ← was int literal
      ("h,help","Show help");
  auto args = cli.parse(argc, argv);
  if (args.count("help") || !args.count("model")) {
    std::cout << cli.help() << "\n\nExample:\n"
              << "  ./gemma_static_server --model ~/gemma3n.litertlm --port 8080\n";
    return 0;
  }
  const std::string model_path = args["model"].as<std::string>();
  const std::string host       = args["host"].as<std::string>();
  const int         port       = args["port"].as<int>();

  httplib::Server srv;
  srv.Post("/v1/chat/completions",
           [&](const httplib::Request& rq, httplib::Response& rs) {
    try {
      const json body = json::parse(rq.body);
      if (body.contains("temperature")) SetGlobalTemperature(body["temperature"].get<float>());

      if (!body.contains("messages") || !body["messages"].is_array())
        throw std::invalid_argument("`messages` array is required");

      const std::string prompt = body["messages"].back()["content"];
      int max_tokens = body.value("max_tokens", -1);

      const auto& eng = EngineHolder::get(model_path);
      std::string answer = generate(eng, prompt, max_tokens);

      rs.set_content(json{
        {"id","chatcmpl-local-0"},
        {"object","chat.completion"},
        {"choices",{{
          {"index",0},{"finish_reason","stop"},
          {"message",{{"role","assistant"},{"content",answer}}}
        }}},
        {"usage", {}}
      }.dump(), "application/json");

    } catch (const std::invalid_argument& ex) {
      rs.status = 400;
      rs.set_content(mk_err(400,"invalid_request_error",ex.what()).dump(),
                     "application/json");
    } catch (const absl::Status& st) {
      rs.status = 500;
      rs.set_content(mk_err(500,"engine_error",st.message()).dump(),
                     "application/json");
    } catch (const std::exception& ex) {
      rs.status = 500;
      rs.set_content(mk_err(500,"internal_error",ex.what()).dump(),
                     "application/json");
    }
  });

  srv.set_exception_handler([](auto, auto& res, std::exception_ptr) {
    res.status = 500;
    res.set_content(mk_err(500,"internal_error","Unhandled exception").dump(),
                    "application/json");
  });

  std::printf("⇢  http://%s:%d/v1/chat/completions  (model: %s)\n",
              host.c_str(), port, model_path.c_str());
  if (!srv.listen(host, port)) {
    std::fprintf(stderr,"Cannot bind %s:%d\n", host.c_str(), port);
    return 1;
  }
}
