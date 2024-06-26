#include "../ai/network.hpp"

#include <iostream>

#include "../external/json/include/nlohmann/json.hpp"
#include "../external/cpp-httplib/httplib.h"

using json = nlohmann::json;
RLAgent agent;

struct FrameInput {

    long int mario_x;
    long int mario_y;
    long int n_enemies;
    long int timer; 
    long int e1_x; 
    long int e1_y;
    long int e2_x; 
    long int e2_y;
    long int e3_x; 
    long int e3_y;
    long int e4_x; 
    long int e4_y;
    long int e5_x; 
    long int e5_y;
    long int is_alive;
};

void set_cors_headers(httplib::Response &res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
    res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    // Include any other headers as needed
}

void handle_options(const httplib::Request &req, httplib::Response &res) {
    set_cors_headers(res);
    res.status = 204; // No Content
}

void handle_post(const httplib::Request &req, httplib::Response &res) {

    set_cors_headers(res); // Set CORS headers for the POST response

    // Parse the request body as JSON
    auto j = json::parse(req.body);

    FrameInput input;

    // Extract data from the JSON object
    input.mario_x = j["mario_x"].get<long int>();
    input.mario_y = j["mario_y"].get<long int>();
    input.n_enemies = j["n_enemies"].get<long int>();
    input.timer = j["timer"].get<long int>();
    input.e1_x = j["e1_x"].get<long int>();
    input.e1_y = j["e1_y"].get<long int>();
    input.e2_y = j["e2_y"].get<long int>();
    input.e2_y = j["e2_y"].get<long int>();
    input.e3_y = j["e3_y"].get<long int>();
    input.e3_y = j["e3_y"].get<long int>();
    input.e4_y = j["e4_y"].get<long int>();
    input.e4_y = j["e4_y"].get<long int>();
    input.e5_y = j["e5_y"].get<long int>();
    input.e5_y = j["e5_y"].get<long int>();
    input.is_alive = j["is_alive"].get<long int>();

    uint64_t prediction = agent.predict(
            input.mario_x,
            input.mario_y,
            input.n_enemies,
            input.timer,
            input.e1_x,
            input.e1_y,
            input.e2_x,
            input.e2_y,
            input.e3_x,
            input.e3_y,
            input.e4_x,
            input.e4_y,
            input.e5_x,
            input.e5_y,
            input.is_alive       
        );


    // Create a response JSON object
    json response_json;
    response_json["action"] = prediction;

    // Send the response JSON as a string
    res.set_content(response_json.dump(), "application/json");
}

int main(void) {
    httplib::Server svr;

    svr.Get("/", [](const httplib::Request&, httplib::Response &res) {
            res.set_content("Hello, World!", "text/plain");
            });

    svr.Post("/post", handle_post);
    svr.Options("/post", handle_options); // Handle pre-flight requests for the POST endpoint

    svr.listen("localhost", 3002);
}
