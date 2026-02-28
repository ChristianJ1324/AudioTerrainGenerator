# audio_terrain

Small C++17 terrain generator that converts a PCM 16-bit WAV file into a deterministic heightmap, normal map, and feature debug strips.

## Layout

- `src/`: generator source
- `third_party/`: vendored `stb_image_write.h`
- `audio_terrain.vcxproj`: standalone Visual Studio project

## Input

- WAV only
- RIFF/WAVE PCM (format 1)
- 16-bit little-endian samples
- mono or stereo
- stereo is downmixed to mono by averaging both channels
- non-audio chunks are skipped safely while parsing

## Feature Extraction

- STFT window: `4096` samples by default (`--window`)
- STFT hop: `1024` samples by default (`--hop`)
- Window function: Hann
- Per frame features:
  - RMS energy
  - 6 band energies
    - sub: 20-60 Hz
    - bass: 60-250 Hz
    - low-mid: 250-500 Hz
    - mid: 500-2000 Hz
    - high-mid: 2000-6000 Hz
    - high: 6000-20000 Hz (clamped by Nyquist)
  - spectral centroid (normalized by Nyquist)
  - onset strength from positive spectral flux
- Each feature stream is smoothed with a short moving average and normalized independently to keep the terrain mapping stable.
- After mapping frames to terrain columns, the per-column features are blurred across neighboring columns to avoid narrow vertical seams.

## Feature To Terrain Mapping

- Song time maps directly to terrain `x`; each terrain column averages the feature frames that fall inside that slice.
- Base terrain is 2D value-noise fBm.
- Sub + bass drive mountain uplift and ridge frequency.
- Low-mid + mid steer medium-scale warping and erosion iterations.
- RMS drives the optional rain-carve strength.
- High-mid + high add small-scale roughness.
- Spectral centroid biases a gentle north/south slope.
- Thermal erosion and rain carving are skipped with `--no_erosion`.

## Outputs

All PNG outputs are written into an `out/` folder under the current working directory.

- `out/features.png`: time-vs-band debug chart (stacked per band)
- `out/height.png`: 16-bit grayscale PNG heightmap
- `out/normal.png`: 8-bit RGB normal map
- `out/preview.png`: simple CPU-shaded terrain preview
- `out/debug_features.png`: extra grayscale feature charts when `--export_debug` is enabled

## Build

Visual Studio / MSBuild:

```powershell
msbuild .\audio_terrain\audio_terrain.vcxproj /p:Configuration=Release /p:Platform=x64
```

The executable is written to `audio_terrain\bin\Release\audio_terrain.exe`.

## Run

From the repo root:

```powershell
.\audio_terrain\bin\Release\audio_terrain.exe input.wav
```

Examples:

```powershell
.\audio_terrain\bin\Release\audio_terrain.exe input.wav --size 1024 --seed 42 --export_debug
.\audio_terrain\bin\Release\audio_terrain.exe input.wav --no_erosion
```

The generated PNGs are written to `out/` inside the current working directory.
