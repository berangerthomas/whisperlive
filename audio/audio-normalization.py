import glob
import pyloudnorm as pyln
import soundfile as sf
import os


target_loudness = -22.0
script_dir = os.path.dirname(os.path.abspath(__file__))
for path in glob.glob(os.path.join(script_dir, "proust*.wav")):
    print(f"Normalizing {path}", end="... ")
    data, sr = sf.read(path)
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(data)

    # Calculate the actual normalization applied
    normalization_amount = target_loudness - loudness
    rounded_amount = round(normalization_amount, 1)
    print(f"{rounded_amount:+} LUFS", flush=True)

    normalized_data = pyln.normalize.loudness(data, loudness, target_loudness)
    out_path = path.replace(".wav", f"_norm_{rounded_amount:+}db.wav")
    sf.write(out_path, normalized_data, sr)
