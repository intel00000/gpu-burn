// console.h
#pragma once

#include "gpuburn/platform.h"
#include <string>
#include <vector>

namespace gpuburn {

class ConsoleTUI final {
  public:
    ConsoleTUI() = default;
    explicit ConsoleTUI(int block_lines);
    ~ConsoleTUI() noexcept;

    ConsoleTUI(const ConsoleTUI &) = delete;
    ConsoleTUI &operator=(const ConsoleTUI &) = delete;
    ConsoleTUI(ConsoleTUI &&) = delete;
    ConsoleTUI &operator=(ConsoleTUI &&) = delete;

    void configure(int block_lines);

    // Draw up to block_lines lines; if fewer provided, remaining lines are
    // blank.
    void draw(const std::vector<std::string> &lines);

    bool is_interactive() const noexcept { return interactive_; }
    int block_lines() const noexcept { return block_lines_; }

  private:
    void start();
    void stop() noexcept;

    bool interactive_ = false; // TTY + ANSI enabled
    bool started_ = false;
    bool first_draw_ = true;
    int block_lines_ = 0;
};

} // namespace gpuburn
