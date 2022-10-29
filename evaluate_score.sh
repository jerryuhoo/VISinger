echo "Scoring"


_gt_wavscp="/home/yyu479/VISinger_data/wav_dump_16k"
_dir="evaluate"
_gen_wavdir="singing_out"

if [ ! -d "singing_gt" ] ; then
    echo "copy gt"
    mkdir -p "singing_gt"
    python normalize_wav.py
fi

# Objective Evaluation - MCD
echo "Begin Scoring for MCD metrics on ${dset}, results are written under ${_dir}/MCD_res"

mkdir -p "${_dir}/MCD_res"
python evaluate/evaluate_mcd.py \
    ${_gen_wavdir} \
    ${_gt_wavscp} \
    --outdir "${_gen_wavdir}/MCD_res"

# Objective Evaluation - log-F0 RMSE
echo "Begin Scoring for F0 related metrics on ${dset}, results are written under ${_dir}/F0_res"

mkdir -p "${_dir}/F0_res"
python evaluate/evaluate_f0.py \
    ${_gen_wavdir} \
    ${_gt_wavscp} \
    --outdir "${_gen_wavdir}/F0_res"

# Objective Evaluation - semitone ACC
echo "Begin Scoring for SEMITONE related metrics on ${dset}, results are written under ${_dir}/SEMITONE_res"

mkdir -p "${_dir}/SEMITONE_res"
python evaluate/evaluate_semitone.py \
    ${_gen_wavdir} \
    ${_gt_wavscp} \
    --outdir "${_gen_wavdir}/SEMITONE_res"

    # Objective Evaluation - VUV error
echo "Begin Scoring for VUV related metrics on ${dset}, results are written under ${_dir}/VUV_res"

mkdir -p "${_dir}/VUV_res"
python evaluate/evaluate_vuv.py \
    ${_gen_wavdir} \
    ${_gt_wavscp} \
    --outdir "${_gen_wavdir}/VUV_res"
