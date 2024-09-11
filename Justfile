test-units:
    cargo test

alias t := test-units

release tag:
    git tag {{tag}}
    cargo publish -p mzdeisotope
    cargo publish -p mzdeisotope-map
    cargo publish -p mzdeisotoper