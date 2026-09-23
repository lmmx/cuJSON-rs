#ifndef CUJSON_ERROR_H
#define CUJSON_ERROR_H

// Thrown by the upstream parsers in place of the process-killing exit()
// calls they shipped with. Caught at the C ABI boundary (task 03's
// capi_*.cu); never allowed to cross extern "C".
enum cujson_error_code {
    CUJSON_ERR_UTF8 = 1,
    CUJSON_ERR_UNBALANCED = 2,
    CUJSON_ERR_INTERNAL = 5,
};

struct cujson_error {
    int code;
};

#endif // CUJSON_ERROR_H
