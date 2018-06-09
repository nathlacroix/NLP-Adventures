env_dir=./env
python_version="3.6.1"
python_path=$(which python)

if [ ! -d "$env_dir" ]; then
    python -m pip install --user virtualenv
    python -m virtualenv \
        --system-site-packages \
        --python=$python_path \
        "$env_dir"
fi

source "$env_dir/bin/activate"
