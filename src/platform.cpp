#include "gpuburn/platform.h"

#include <cstdarg>
#include <cstdio>

namespace gpuburn {

namespace {
static std::atomic<bool> *g_running_ptr = nullptr;
}

#ifdef _WIN32
#include <io.h>
#include <windows.h>

static BOOL WINAPI console_handler(DWORD type) {
    if (!g_running_ptr)
        return FALSE;

    switch (type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        g_running_ptr->store(false, std::memory_order_relaxed);
        return TRUE;
    default:
        return FALSE;
    }
}

void install_signal_handlers(std::atomic<bool> &running) {
    g_running_ptr = &running;
    SetConsoleCtrlHandler(console_handler, TRUE);
}

bool enable_ansi() {
    if (!_isatty(_fileno(stdout)))
        return false;

    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h == INVALID_HANDLE_VALUE)
        return false;

    DWORD mode = 0;
    if (!GetConsoleMode(h, &mode))
        return false;

    mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    return SetConsoleMode(h, mode) != 0;
}

#else
#include <signal.h>
#include <unistd.h>

static void sig_handler(int) {
    if (g_running_ptr)
        g_running_ptr->store(false, std::memory_order_relaxed);
}

void install_signal_handlers(std::atomic<bool> &running) {
    g_running_ptr = &running;

    struct sigaction sa{};
    sa.sa_handler = sig_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
}

bool enable_ansi() { return isatty(fileno(stdout)) != 0; }
#endif

std::string sformat(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);

    va_list ap2;
    va_copy(ap2, ap);
    int n = std::vsnprintf(nullptr, 0, fmt, ap2);
    va_end(ap2);

    std::string s;
    if (n > 0) {
        s.resize((size_t)n);
        std::vsnprintf(s.data(), (size_t)n + 1, fmt, ap);
    }

    va_end(ap);
    return s;
}

} // namespace gpuburn
