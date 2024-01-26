RAW=$1
OUT=$2

# export MZDEIOSTOPE_BLOSC_ZSTD=9

cargo b --features mzmlb --release

msconvert --mzML $RAW -o- | cargo r --features mzmlb --release -- \
    - -g 1 -s 10 -a glycopeptide \
    --msn-score-threshold 5 --msn-isotopic-model peptide \
    -o $OUT