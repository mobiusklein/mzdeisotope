RAW=$1
OUT=$2

# export MZDEIOSTOPE_BLOSC_ZSTD=9
export RUST_LOG=mzdeisotoper::proc=debug,info

cargo b --release

msconvert --mzML $RAW -o- | cargo r --release -- \
    - -g 1 -s 10 -a glycopeptide \
    --msn-score-threshold 5 -A peptide -A glycopeptide \
    -o $OUT