RAW=$1
OUT=$2

msconvert --mzML  $RAW -o- | cargo r --release -- \
    - -g 1 -s 10 -a glycopeptide \
    --msn-score-threshold 5 --msn-isotopic-model peptide \
    -o $OUT