// loadfile.h
#ifndef LOADFILE_H
#define LOADFILE_H

#include <string>
#include "cujson_types.h"

// load_file.cu's loadJSON/loadJSONLines_* are file-scope (static) - this
// header only brings in cuJSONInput/cuJSONLinesInput for callers that
// don't need file loading (this TU is compiled into both cujson.h's and
// cujsonlines.h's unity build, so external declarations here would
// collide with the static definitions in load_file.cu).

#endif // LOADFILE_H
