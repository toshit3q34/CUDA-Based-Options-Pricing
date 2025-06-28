#pragma once
#include <chrono>
#include <functional>
#include <iostream>
#include <string>

class Timer {
private:
    std::string label;
    std::chrono::high_resolution_clock::time_point start;
    
public:
    Timer(const std::string& label = "") : label(label), start(std::chrono::high_resolution_clock::now()) {}

    double getDuration() const {
      auto end = std::chrono::high_resolution_clock::now();
      return std::chrono::duration<double>(end - start).count();
    }

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        std::cout << label << " took " << duration << " seconds." << std::endl;
    }
};