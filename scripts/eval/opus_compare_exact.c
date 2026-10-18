/* libopus src/opus_compare.c with full-precision output.

   This file includes opus_compare.c unchanged. It only replaces fprintf by
   a function that prints every "%f" and "%.1f" as "%.17g", so that the
   internal weighted error (a double) and the quality Q (a float, promoted
   to double) come out exactly. tests/eval_opus_compare.rs reads them back
   and compares them bit for bit with the Rust port.

   Build it with scripts/eval/check-opus-compare.sh, which also checks that
   the arithmetic of the wrapped program equals that of the plain one. */
#include <stdarg.h>
#include <stdio.h>

static int exact_fprintf(FILE *f, const char *fmt, ...);
#define fprintf exact_fprintf
#include "opus_compare.c" /* found with -I<libopus>/src */
#undef fprintf

static int exact_fprintf(FILE *f, const char *fmt, ...) {
  char out[512];
  size_t o = 0;
  const char *p;
  va_list ap;
  int r;
  for (p = fmt; *p && o + 6 < sizeof out; p++) {
    if (p[0] == '%' && p[1] == '%') {
      out[o++] = '%';
      out[o++] = '%';
      p += 1;
    } else if (p[0] == '%' && p[1] == 'f') {
      memcpy(out + o, "%.17g", 5);
      o += 5;
      p += 1;
    } else if (p[0] == '%' && p[1] == '.' && p[2] == '1' && p[3] == 'f') {
      memcpy(out + o, "%.17g", 5);
      o += 5;
      p += 3;
    } else {
      out[o++] = *p;
    }
  }
  out[o] = 0;
  va_start(ap, fmt);
  r = vfprintf(f, out, ap);
  va_end(ap);
  return r;
}
