#ifndef JSON_LINES_PARSE_H
#define JSON_LINES_PARSE_H

#include <string>
#include "cujson_types.h"

// Function prototype for json_lines parse
// cuJSONResult standard_json_parse(uint8_t* h_jsonContent);
namespace cujson_lines {
cuJSONResult parse_json_lines(cuJSONLinesInput input);
} // namespace cujson_lines

#endif // JSON_LINES_PARSE_H
