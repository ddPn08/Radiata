# python3 executable
if [[ -z "${python_cmd}" ]]
then
    python_cmd="python3"
fi

# git executable
if [[ -z "${GIT}" ]]
then
    export GIT="git"
fi

# python3 venv without trailing slash
if [[ -z "${venv_dir}" ]]
then
    venv_dir="venv"
fi

if [[ -z "${LAUNCH_SCRIPT}" ]]
then
    LAUNCH_SCRIPT="launch.py"
fi

# this script cannot be run as root by default
can_run_as_root=0
delimiter="################################################################"

printf "\n%s\n" "${delimiter}"
printf "Create and activate python venv"
printf "\n%s\n" "${delimiter}"
if [[ ! -d "${venv_dir}" ]]
then
    "${python_cmd}" -m venv "${venv_dir}"
    first_launch=1
fi
# shellcheck source=/dev/null
if [[ -f "${venv_dir}"/bin/activate ]]
then
    source "${venv_dir}"/bin/activate
else
    printf "\n%s\n" "${delimiter}"
    printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
    printf "\n%s\n" "${delimiter}"
    exit 1
fi

if [[ ! -z "${ACCELERATE}" ]] && [ ${ACCELERATE}="True" ] && [ -x "$(command -v accelerate)" ]
then
    printf "\n%s\n" "${delimiter}"
    printf "Accelerating launch.py..."
    printf "\n%s\n" "${delimiter}"
    exec accelerate launch --num_cpu_threads_per_process=6 "${LAUNCH_SCRIPT}" "$@"
else
    printf "\n%s\n" "${delimiter}"
    printf "Launching launch.py..."
    printf "\n%s\n" "${delimiter}"
    exec "${python_cmd}" -u "${LAUNCH_SCRIPT}" "$@"
fi