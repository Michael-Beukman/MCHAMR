export PYTHONPATH=`pwd`/external/gym-pcgrl:`pwd`:`pwd`/../Evocraft-py:external/Evocraft-py:external/ruck

if [ -f $1 ]; then
    nice -n 0 python -u  "$@"
else
    nice -n 0 python -u  ../"$@"
fi

