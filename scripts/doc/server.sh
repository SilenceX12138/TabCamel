cd docs/wiki
mkdir -p _static _templates
make clean
make html
python -m http.server --directory _build/html
