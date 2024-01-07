{
  "targets": [{
      "target_name": "testaddon",
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "sources": [
        "cpp-code/main.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "ldflags": [
        "-Ldll",
        "-lstack_dll.dll"
      ],
      "defines": ["NAPI_DISABLE_CPP_EXCEPTIONS"]
    }]
}
