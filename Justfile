test-units:
    cargo nextest run

test-coverage:
    cargo llvm-cov nextest --lib --tests --html

alias t := test-units

release tag:
    git tag {{tag}}
    cargo publish -p mzdeisotope
    cargo publish -p mzdeisotope-map
    cargo publish -p mzdeisotoper