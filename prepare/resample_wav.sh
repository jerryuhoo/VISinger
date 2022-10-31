OPENCPOP=/home/yyu479/svs/data/Opencpop/
fs=24000
output=/home/yyu479/VISinger_data/wav_dump_24k
mkdir -p ${output}
python resample_wav.py ${OPENCPOP} \
        --wav_dumpdir ${output} \
        --sr ${fs}