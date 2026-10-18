#!/usr/bin/env bash
# Download and unpack a corpus of the evaluation tools, or check one that is
# already there.
#
#   scripts/eval/fetch-corpus.sh [--dest DIR] [--yes] rfc8251 [URL]
#   scripts/eval/fetch-corpus.sh [--dest DIR] [--yes] tsp URL
#   scripts/eval/fetch-corpus.sh [--dest DIR] --verify rfc8251|tsp
#
# Corpora:
#   rfc8251  The test vectors of RFC 8251 (74,624,664 bytes), from
#            https://opus-codec.org/docs/opus_testvectors-rfc8251.tar.gz
#            (libopus README). The SHA-256 of the archive is the one in
#            libopus's releases.sha2. After unpacking, the SHA-1 of every
#            .bit, .dec and m.dec file is checked against the list in
#            draft-ietf-codec-opus-update (the draft of RFC 8251). Layout:
#            DEST/rfc8251/opus_newvectors/testvectorNN{.bit,.dec,m.dec}.
#   tsp      The TSP speech database at 48 kHz, McGill University
#            (253,816,919 bytes, simplified BSD licence). The page is
#            https://www.mmsp.ece.mcgill.ca/Documents/Data/ ; pass the URL
#            of the 48 kHz archive (48k.zip). No checksum is published. The
#            size and SHA-256 below were recorded at the first download on
#            2026-10-18 (trust on first use). Layout:
#            DEST/tsp48k/48k/<talker>/<file>.wav, with 48k/Licence.txt.
#
# DEST is --dest, else OPUS1_CORPUS, else <repository>/corpus (git-ignored).
#
# The script asks before it downloads (unless --yes), downloads with curl
# into a new empty directory, checks the size and the SHA-256, refuses
# archive entries with absolute paths or "..", unpacks with tar or unzip
# into another new empty directory, and moves the result into place. It
# never runs anything from an archive. It records the SHA-256 of the archive
# in DEST/SHA256SUMS.
set -euo pipefail

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; }

DEST=""
YES=0
VERIFY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dest) DEST=$2; shift 2 ;;
    --yes) YES=1; shift ;;
    --verify) VERIFY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    *) break ;;
  esac
done
NAME=${1:-}
URL=${2:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DEST=${DEST:-${OPUS1_CORPUS:-$REPO/corpus}}

case "$NAME" in
  rfc8251)
    SUBDIR=rfc8251
    ARCHIVE=opus_testvectors-rfc8251.tar.gz
    DEFAULT_URL=https://opus-codec.org/docs/opus_testvectors-rfc8251.tar.gz
    SIZE=74624664
    SHA256=6b26a22f9ba87b2b836906a9bb7afec5f8e54d49553b1200382520ee6fedfa55
    ;;
  tsp)
    SUBDIR=tsp48k
    ARCHIVE=48k.zip
    DEFAULT_URL=""
    SIZE=253816919
    SHA256=9cfb3a3a13014c8ff90770a5d1923f376da73ac927b9100f09826f60cf06cf43
    ;;
  *) usage >&2; exit 2 ;;
esac

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
sha1() {
  if command -v sha1sum >/dev/null; then sha1sum "$1" | cut -d' ' -f1; else shasum -a 1 "$1" | cut -d' ' -f1; fi
}
size_of() { wc -c < "$1" | tr -d ' '; }

# The SHA-1 of the RFC 8251 files (draft-ietf-codec-opus-update, as shipped
# in the libopus git repository, doc/draft-ietf-codec-opus-update.xml).
RFC8251_SHA1="
e49b2862ceec7324790ed8019eb9744596d5be01 testvector01.bit
b809795ae1bcd606049d76de4ad24236257135e0 testvector02.bit
e0c4ecaeab44d35a2f5b6575cd996848e5ee2acc testvector03.bit
a0f870cbe14ebb71fa9066ef3ee96e59c9a75187 testvector04.bit
9b3d92b48b965dfe9edf7b8a85edd4309f8cf7c8 testvector05.bit
28e66769ab17e17f72875283c14b19690cbc4e57 testvector06.bit
bacf467be3215fc7ec288f29e2477de1192947a6 testvector07.bit
ddbe08b688bbf934071f3893cd0030ce48dba12f testvector08.bit
3932d9d61944dab1201645b8eeaad595d5705ecb testvector09.bit
521eb2a1e0cc9c31b8b740673307c2d3b10c1900 testvector10.bit
6bc8f3146fcb96450c901b16c3d464ccdf4d5d96 testvector11.bit
338c3f1b4b97226bc60bc41038becbc6de06b28f testvector12.bit
f5ef93884da6a814d311027918e9afc6f2e5c2c8 testvector01.dec
48ac1ff1995250a756e1e17bd32acefa8cd2b820 testvector02.dec
d15567e919db2d0e818727092c0af8dd9df23c95 testvector03.dec
1249dd28f5bd1e39a66fd6d99449dca7a8316342 testvector04.dec
b85675d81deef84a112c466cdff3b7aaa1d2fc76 testvector05.dec
55f0b191e90bfa6f98b50d01a64b44255cb4813e testvector06.dec
61e8b357ab090b1801eeb578a28a6ae935e25b7b testvector07.dec
a58539ee5321453b2ddf4c0f2500e856b3966862 testvector08.dec
bb96aad2cde188555862b7bbb3af6133851ef8f4 testvector09.dec
1b6cdf0413ac9965b16184b1bea129b5c0b2a37a testvector10.dec
b1fff72b74666e3027801b29dbc48b31f80dee0d testvector11.dec
98e09bbafed329e341c3b4052e9c4ba5fc83f9b1 testvector12.dec
1e7d984ea3fbb16ba998aea761f4893fbdb30157 testvector01m.dec
48ac1ff1995250a756e1e17bd32acefa8cd2b820 testvector02m.dec
d15567e919db2d0e818727092c0af8dd9df23c95 testvector03m.dec
1249dd28f5bd1e39a66fd6d99449dca7a8316342 testvector04m.dec
d70b0bad431e7d463bc3da49bd2d49f1c6d0a530 testvector05m.dec
6ac1648c3174c95fada565161a6c78bdbe59c77d testvector06m.dec
fc5e2f709693738324fb4c8bdc0dad6dda04e713 testvector07m.dec
aad2ba397bf1b6a18e8e09b50e4b19627d479f00 testvector08m.dec
6feb7a7b9d7cdc1383baf8d5739e2a514bd0ba08 testvector09m.dec
1b6cdf0413ac9965b16184b1bea129b5c0b2a37a testvector10m.dec
fd3d3a7b0dfbdab98d37ed9aa04b659b9fefbd18 testvector11m.dec
98e09bbafed329e341c3b4052e9c4ba5fc83f9b1 testvector12m.dec
"

# Check the size and SHA-256 of an archive. Returns 1 on a mismatch.
check_archive() {
  local file=$1 got_size got_sha
  got_size=$(size_of "$file")
  if [ "$got_size" != "$SIZE" ]; then
    echo "size mismatch: $file has $got_size bytes, expected $SIZE" >&2
    return 1
  fi
  got_sha=$(sha256 "$file")
  if [ "$got_sha" != "$SHA256" ]; then
    echo "SHA-256 mismatch: $file has $got_sha, expected $SHA256" >&2
    return 1
  fi
  echo "ok: $file ($got_size bytes, SHA-256 $got_sha)"
}

# Check the unpacked files.
check_unpacked() {
  local root=$1 bad=0 n line sum file got
  case "$NAME" in
    rfc8251)
      local dir=$root/opus_newvectors
      n=0
      while read -r sum file; do
        [ -n "$sum" ] || continue
        if [ ! -f "$dir/$file" ]; then
          echo "missing: $dir/$file" >&2; bad=1; continue
        fi
        got=$(sha1 "$dir/$file")
        if [ "$got" != "$sum" ]; then
          echo "SHA-1 mismatch: $dir/$file has $got, expected $sum" >&2; bad=1
        fi
        n=$((n + 1))
      done <<< "$RFC8251_SHA1"
      [ "$bad" = 0 ] && echo "ok: the SHA-1 of all $n files in $dir"
      ;;
    tsp)
      local dir=$root/48k
      if [ ! -f "$dir/Licence.txt" ]; then echo "missing: $dir/Licence.txt" >&2; bad=1; fi
      n=$(find "$dir" -name '*.wav' | wc -l | tr -d ' ')
      if [ "$n" != 1444 ]; then echo "expected 1444 WAV files in $dir, found $n" >&2; bad=1; else echo "ok: $n WAV files and Licence.txt in $dir"; fi
      ;;
  esac
  return $bad
}

# Refuse archive entries with an absolute path or a ".." component.
check_entries() {
  local file=$1 list
  case "$file" in
    *.tar.gz) list=$(tar -tzf "$file") ;;
    *.zip) list=$(unzip -Z1 "$file") ;;
  esac
  if printf '%s\n' "$list" | grep -Eq '^/|(^|/)\.\.(/|$)'; then
    echo "refused: $file has entries with an absolute path or '..'" >&2
    return 1
  fi
}

TARGET=$DEST/$SUBDIR

if [ "$VERIFY" = 1 ]; then
  status=0
  if [ -f "$TARGET/$ARCHIVE" ]; then check_archive "$TARGET/$ARCHIVE" || status=1; else echo "no archive at $TARGET/$ARCHIVE (only the unpacked files are checked)"; fi
  check_unpacked "$TARGET" || status=1
  exit $status
fi

URL=${URL:-$DEFAULT_URL}
if [ -z "$URL" ]; then
  echo "give the URL of the archive: the TSP page is https://www.mmsp.ece.mcgill.ca/Documents/Data/" >&2
  exit 2
fi
case "$URL" in
  https://*|file://*) ;;
  *) echo "refused: only https:// (or file:// for a local copy) URLs" >&2; exit 2 ;;
esac
if [ -e "$TARGET/$ARCHIVE" ]; then
  echo "$TARGET/$ARCHIVE exists already; check it with --verify" >&2
  exit 1
fi

echo "corpus:      $NAME"
echo "source:      $URL"
echo "size:        $SIZE bytes"
echo "SHA-256:     $SHA256"
echo "destination: $TARGET"
if [ "$YES" != 1 ]; then
  printf 'Download it? [y/N] '
  read -r answer
  case "$answer" in y|Y|yes) ;; *) echo "nothing downloaded"; exit 1 ;; esac
fi

mkdir -p "$TARGET"
download=$(mktemp -d "$TARGET/.download.XXXXXX")
unpack=$(mktemp -d "$TARGET/.unpack.XXXXXX")
cleanup() { rm -rf "$download" "$unpack"; }
trap cleanup EXIT

curl -L --fail --proto '=https,file' --max-filesize $((SIZE + 1)) -o "$download/$ARCHIVE" "$URL"
check_archive "$download/$ARCHIVE"
check_entries "$download/$ARCHIVE"
case "$ARCHIVE" in
  *.tar.gz) tar -xzf "$download/$ARCHIVE" -C "$unpack" ;;
  *.zip) unzip -q "$download/$ARCHIVE" -d "$unpack" ;;
esac
check_unpacked "$unpack"

for entry in "$unpack"/* ; do
  name=$(basename "$entry")
  if [ -e "$TARGET/$name" ]; then
    echo "$TARGET/$name exists already; left as it is, the new copy is discarded" >&2
  else
    mv "$entry" "$TARGET/$name"
  fi
done
mv "$download/$ARCHIVE" "$TARGET/$ARCHIVE"

# Record the checksum, replacing an older line for the same archive.
sums=$DEST/SHA256SUMS
touch "$sums"
grep -v "  $SUBDIR/$ARCHIVE\$" "$sums" > "$sums.tmp" || true
echo "$SHA256  $SUBDIR/$ARCHIVE" >> "$sums.tmp"
mv "$sums.tmp" "$sums"
echo "done: $TARGET"
