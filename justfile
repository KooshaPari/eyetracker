# Phenotype-org shared justfile. Imported from phenotype-tooling/just/phenotype.just.
# To override a recipe locally, redefine it after the import.
import? "/Users/kooshapari/CodeProjects/Phenotype/repos/phenotype-tooling/just/phenotype.just"

# Measure code coverage (SSOT: see grade.sh for the canonical command)
coverage:
    cargo llvm-cov --workspace --fail-under-lines 85
