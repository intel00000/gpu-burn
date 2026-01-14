// console.cpp
#include "gpuburn/console.h"

#include <cstdio>

namespace gpuburn {

ConsoleTUI::ConsoleTUI(int block_lines) { configure(block_lines); }

ConsoleTUI::~ConsoleTUI() noexcept { stop(); }

void ConsoleTUI::configure(int block_lines) {
    if (block_lines <= 0) {
        // disable / no-op
        stop();
        block_lines_ = 0;
        return;
    }

    // If already started with a different size, stop and restart.
    const bool need_restart = started_ && (block_lines_ != block_lines);
    if (need_restart)
        stop();
    block_lines_ = block_lines;

    if (!started_)
        start();
}

void ConsoleTUI::start() {
    // block_lines_ already set (may be 0).
    interactive_ = (block_lines_ > 0) ? enable_ansi() : false;
    started_ = true;
    first_draw_ = true;

    if (interactive_) {
        std::printf("\x1b[?25l"); // hide cursor
        std::fflush(stdout);
    }
}

void ConsoleTUI::stop() noexcept {
    if (!started_)
        return;

    if (interactive_) {
        std::printf("\x1b[?25h"); // show cursor
        std::fflush(stdout);
    }

    interactive_ = false;
    started_ = false;
    first_draw_ = true;
}

void ConsoleTUI::draw(const std::vector<std::string> &lines) {
    if (block_lines_ <= 0) {
        for (const auto &ln : lines)
            std::printf("%s\n", ln.c_str());
        std::fflush(stdout);
        return;
    }

    if (!started_)
        start();

    if (!interactive_) {
        for (const auto &ln : lines)
            std::printf("%s\n", ln.c_str());
        std::fflush(stdout);
        return;
    }

    // Move cursor to top of the block after the first draw.
    if (!first_draw_)
        std::printf("\x1b[%dA", block_lines_);
    first_draw_ = false;

    // Rewrite the entire block.
    for (int i = 0; i < block_lines_; ++i) {
        std::printf("\x1b[2K"); // clear entire line
        if (i < static_cast<int>(lines.size())) {
            std::printf("%s", lines[static_cast<size_t>(i)].c_str());
        }
        std::printf("\n");
    }
    std::fflush(stdout);
}

} // namespace gpuburn
