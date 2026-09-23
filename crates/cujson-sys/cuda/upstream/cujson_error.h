#ifndef CUJSON_ERROR_H
#define CUJSON_ERROR_H

// Thrown by the upstream parsers in place of the process-killing exit()
// calls they shipped with. Caught at the C ABI boundary (task 03's
// capi_*.cu); never allowed to cross extern "C".
//
// Namespaced (not a plain enum) because cujson_capi.h's cujson_status
// enum, included in the same translation unit as this header by every
// capi_*.cu, uses the same CUJSON_ERR_* names - a plain enum here would
// collide at global scope.
namespace cujson_err {
constexpr int UTF8 = 1;
constexpr int UNBALANCED = 2;
constexpr int INTERNAL = 5;
constexpr int EMPTY_INPUT = 6;
} // namespace cujson_err

struct cujson_error {
    int code;
};

#endif // CUJSON_ERROR_H
