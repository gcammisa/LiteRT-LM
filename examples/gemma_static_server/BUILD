# examples/gemma_static_server/BUILD
load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "gemma_static_server",
    srcs = ["server.cc"],
    deps = [
        "//runtime/core:engine_impl",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "@com_github_yhirose_cpp_httplib//:cpp_httplib",
        "@com_github_nlohmann_json//:json",
        "@com_github_jarro2783_cxxopts//:cxxopts",   # NEW
    ],
    copts = ["-std=c++20"],
    linkopts = ["-static"],
    features = ["fully_static_link"],
    linkstatic = True,
)
