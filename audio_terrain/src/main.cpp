#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

namespace
{
constexpr float kPi = 3.14159265358979323846f;
constexpr int kBandCount = 6;

struct Config
{
    std::string inputPath;
    int size = 512;
    int windowSize = 4096;
    int hopSize = 1024;
    std::uint32_t seed = 1337u;
    bool disableErosion = false;
    bool exportDebug = false;
};

struct AudioData
{
    int sampleRate = 0;
    std::vector<float> samples;
};

struct FeatureFrame
{
    float rms = 0.0f;
    std::array<float, kBandCount> bands{};
    float centroid = 0.0f;
    float onset = 0.0f;
};

struct ColumnFeatures
{
    float rms = 0.0f;
    std::array<float, kBandCount> bands{};
    float centroid = 0.0f;
    float onset = 0.0f;
};

std::uint16_t ReadLe16(const std::vector<std::uint8_t>& bytes, std::size_t offset)
{
    return static_cast<std::uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
}

std::uint32_t ReadLe32(const std::vector<std::uint8_t>& bytes, std::size_t offset)
{
    return static_cast<std::uint32_t>(bytes[offset]) |
        (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
        (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
        (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

float Clamp01(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

float Lerp(float a, float b, float t)
{
    return a + (b - a) * t;
}

float SmoothStep(float t)
{
    return t * t * (3.0f - 2.0f * t);
}

std::uint32_t Hash(std::uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float HashToUnit(std::uint32_t x, std::uint32_t y, std::uint32_t seed)
{
    const std::uint32_t mixed = Hash(x * 0x9e3779b9u ^ y * 0x85ebca6bu ^ seed);
    return static_cast<float>(mixed & 0x00ffffffu) / static_cast<float>(0x01000000u);
}

float ValueNoise(float x, float y, std::uint32_t seed)
{
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float fx = SmoothStep(x - static_cast<float>(x0));
    const float fy = SmoothStep(y - static_cast<float>(y0));

    const float v00 = HashToUnit(static_cast<std::uint32_t>(x0), static_cast<std::uint32_t>(y0), seed);
    const float v10 = HashToUnit(static_cast<std::uint32_t>(x1), static_cast<std::uint32_t>(y0), seed);
    const float v01 = HashToUnit(static_cast<std::uint32_t>(x0), static_cast<std::uint32_t>(y1), seed);
    const float v11 = HashToUnit(static_cast<std::uint32_t>(x1), static_cast<std::uint32_t>(y1), seed);

    return Lerp(Lerp(v00, v10, fx), Lerp(v01, v11, fx), fy);
}

float Fbm(float x, float y, int octaves, float lacunarity, float gain, std::uint32_t seed)
{
    float amplitude = 0.5f;
    float frequency = 1.0f;
    float sum = 0.0f;
    float norm = 0.0f;

    for (int octave = 0; octave < octaves; ++octave)
    {
        sum += amplitude * ValueNoise(x * frequency, y * frequency, seed + static_cast<std::uint32_t>(octave) * 1013u);
        norm += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }

    return norm > 0.0f ? sum / norm : 0.0f;
}

bool IsPowerOfTwo(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

void PrintUsage()
{
    std::cout
        << "Usage: audio_terrain input.wav [--size 512|1024] [--seed N] [--no_erosion] [--export_debug]\n"
        << "       [--window 4096] [--hop 1024]\n";
}

bool ParseIntArg(const std::string& text, int& value)
{
    try
    {
        std::size_t parsed = 0;
        value = std::stoi(text, &parsed, 10);
        return parsed == text.size();
    }
    catch (...)
    {
        return false;
    }
}

bool ParseUIntArg(const std::string& text, std::uint32_t& value)
{
    try
    {
        std::size_t parsed = 0;
        const unsigned long parsedValue = std::stoul(text, &parsed, 10);
        if (parsed != text.size())
        {
            return false;
        }
        value = static_cast<std::uint32_t>(parsedValue);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool ParseArgs(int argc, char** argv, Config& config)
{
    if (argc < 2)
    {
        return false;
    }

    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h")
        {
            return false;
        }
        if (!arg.empty() && arg[0] != '-')
        {
            if (!config.inputPath.empty())
            {
                std::cerr << "Only one input WAV file can be specified.\n";
                return false;
            }
            config.inputPath = arg;
            continue;
        }
        if (arg == "--no_erosion")
        {
            config.disableErosion = true;
            continue;
        }
        if (arg == "--export_debug")
        {
            config.exportDebug = true;
            continue;
        }
        if ((arg == "--size" || arg == "--window" || arg == "--hop" || arg == "--seed") && index + 1 >= argc)
        {
            std::cerr << "Missing value for " << arg << ".\n";
            return false;
        }
        if (arg == "--size")
        {
            int value = 0;
            if (!ParseIntArg(argv[++index], value) || (value != 512 && value != 1024))
            {
                std::cerr << "--size must be 512 or 1024.\n";
                return false;
            }
            config.size = value;
            continue;
        }
        if (arg == "--window")
        {
            int value = 0;
            if (!ParseIntArg(argv[++index], value) || !IsPowerOfTwo(value) || value < 256)
            {
                std::cerr << "--window must be a power of two >= 256.\n";
                return false;
            }
            config.windowSize = value;
            continue;
        }
        if (arg == "--hop")
        {
            int value = 0;
            if (!ParseIntArg(argv[++index], value) || value <= 0)
            {
                std::cerr << "--hop must be > 0.\n";
                return false;
            }
            config.hopSize = value;
            continue;
        }
        if (arg == "--seed")
        {
            std::uint32_t value = 0;
            if (!ParseUIntArg(argv[++index], value))
            {
                std::cerr << "--seed must be an unsigned integer.\n";
                return false;
            }
            config.seed = value;
            continue;
        }
        std::cerr << "Unknown argument: " << arg << "\n";
        return false;
    }

    if (config.inputPath.empty())
    {
        std::cerr << "An input WAV file is required.\n";
        return false;
    }
    if (config.hopSize > config.windowSize)
    {
        std::cerr << "--hop must not exceed --window.\n";
        return false;
    }
    return true;
}

bool LoadWav(const std::string& path, AudioData& audio, std::string& error)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        error = "Failed to open input file.";
        return false;
    }

    input.seekg(0, std::ios::end);
    const std::streamoff length = input.tellg();
    input.seekg(0, std::ios::beg);
    if (length < 44)
    {
        error = "File is too small to be a valid WAV.";
        return false;
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(length));
    if (!input)
    {
        error = "Failed to read WAV file contents.";
        return false;
    }

    if (std::string(reinterpret_cast<const char*>(bytes.data()), 4) != "RIFF" ||
        std::string(reinterpret_cast<const char*>(bytes.data() + 8), 4) != "WAVE")
    {
        error = "Input is not a RIFF/WAVE file.";
        return false;
    }

    bool sawFmt = false;
    std::uint16_t audioFormat = 0;
    std::uint16_t channels = 0;
    std::uint16_t bitsPerSample = 0;
    std::uint32_t sampleRate = 0;
    std::vector<std::uint8_t> pcmBytes;

    std::size_t offset = 12;
    while (offset + 8 <= bytes.size())
    {
        const std::string chunkId(reinterpret_cast<const char*>(bytes.data() + offset), 4);
        const std::uint32_t chunkSize = ReadLe32(bytes, offset + 4);
        const std::size_t chunkData = offset + 8;
        if (chunkData + chunkSize > bytes.size())
        {
            error = "Encountered a truncated WAV chunk.";
            return false;
        }

        if (chunkId == "fmt ")
        {
            if (chunkSize < 16)
            {
                error = "Unsupported fmt chunk.";
                return false;
            }
            audioFormat = ReadLe16(bytes, chunkData);
            channels = ReadLe16(bytes, chunkData + 2);
            sampleRate = ReadLe32(bytes, chunkData + 4);
            bitsPerSample = ReadLe16(bytes, chunkData + 14);
            sawFmt = true;
        }
        else if (chunkId == "data")
        {
            pcmBytes.insert(pcmBytes.end(), bytes.begin() + static_cast<std::ptrdiff_t>(chunkData),
                bytes.begin() + static_cast<std::ptrdiff_t>(chunkData + chunkSize));
        }

        offset = chunkData + chunkSize + (chunkSize & 1u);
    }
    if (!sawFmt)
    {
        error = "WAV is missing a fmt chunk.";
        return false;
    }
    if (audioFormat != 1)
    {
        error = "Only PCM WAV files are supported.";
        return false;
    }
    if ((channels != 1 && channels != 2) || bitsPerSample != 16 || sampleRate == 0)
    {
        error = "Expected 16-bit PCM mono or stereo WAV data.";
        return false;
    }
    if (pcmBytes.empty())
    {
        error = "WAV is missing audio data.";
        return false;
    }

    const std::size_t frameStride = static_cast<std::size_t>(channels) * sizeof(std::int16_t);
    const std::size_t frameCount = pcmBytes.size() / frameStride;
    if (frameCount == 0)
    {
        error = "WAV data chunk is empty.";
        return false;
    }

    audio.sampleRate = static_cast<int>(sampleRate);
    audio.samples.resize(frameCount);

    for (std::size_t frame = 0; frame < frameCount; ++frame)
    {
        if (channels == 1)
        {
            const std::size_t offsetBytes = frame * 2;
            const std::int16_t sample = static_cast<std::int16_t>(ReadLe16(pcmBytes, offsetBytes));
            audio.samples[frame] = static_cast<float>(sample) / 32768.0f;
        }
        else
        {
            const std::size_t base = frame * 4;
            const std::int16_t leftSample = static_cast<std::int16_t>(ReadLe16(pcmBytes, base));
            const std::int16_t rightSample = static_cast<std::int16_t>(ReadLe16(pcmBytes, base + 2));
            const float left = static_cast<float>(leftSample) / 32768.0f;
            const float right = static_cast<float>(rightSample) / 32768.0f;
            audio.samples[frame] = 0.5f * (left + right);
        }
    }

    return true;
}

void FFT(std::vector<std::complex<float>>& values)
{
    const std::size_t count = values.size();
    std::size_t j = 0;
    for (std::size_t i = 1; i < count; ++i)
    {
        std::size_t bit = count >> 1;
        while (j & bit)
        {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j)
        {
            std::swap(values[i], values[j]);
        }
    }

    for (std::size_t len = 2; len <= count; len <<= 1)
    {
        const float angle = -2.0f * kPi / static_cast<float>(len);
        const std::complex<float> step(std::cos(angle), std::sin(angle));
        for (std::size_t i = 0; i < count; i += len)
        {
            std::complex<float> omega(1.0f, 0.0f);
            const std::size_t half = len >> 1;
            for (std::size_t k = 0; k < half; ++k)
            {
                const std::complex<float> even = values[i + k];
                const std::complex<float> odd = values[i + k + half] * omega;
                values[i + k] = even + odd;
                values[i + k + half] = even - odd;
                omega *= step;
            }
        }
    }
}

std::vector<float> MovingAverage(const std::vector<float>& values, int radius)
{
    if (values.empty())
    {
        return {};
    }

    const int count = static_cast<int>(values.size());
    std::vector<float> prefix(values.size() + 1, 0.0f);
    for (int index = 0; index < count; ++index)
    {
        prefix[index + 1] = prefix[index] + values[index];
    }

    std::vector<float> result(values.size(), 0.0f);
    for (int index = 0; index < count; ++index)
    {
        const int begin = std::max(0, index - radius);
        const int end = std::min(count, index + radius + 1);
        const float sum = prefix[end] - prefix[begin];
        result[index] = sum / static_cast<float>(end - begin);
    }
    return result;
}

std::vector<FeatureFrame> AnalyzeAudio(const AudioData& audio, const Config& config)
{
    const int windowSize = config.windowSize;
    const int hopSize = config.hopSize;
    const int sampleCount = static_cast<int>(audio.samples.size());
    const int frameCount = std::max(1, static_cast<int>(std::ceil(std::max(0, sampleCount - windowSize) / static_cast<double>(hopSize))) + 1);
    const int spectrumBins = windowSize / 2 + 1;
    const float binHz = static_cast<float>(audio.sampleRate) / static_cast<float>(windowSize);
    const float nyquist = 0.5f * static_cast<float>(audio.sampleRate);

    std::vector<float> hann(windowSize, 0.0f);
    for (int i = 0; i < windowSize; ++i)
    {
        hann[i] = 0.5f - 0.5f * std::cos((2.0f * kPi * static_cast<float>(i)) / static_cast<float>(windowSize - 1));
    }

    const std::array<std::pair<float, float>, kBandCount> bands = {
        std::pair<float, float>{20.0f, 60.0f},
        std::pair<float, float>{60.0f, 250.0f},
        std::pair<float, float>{250.0f, 500.0f},
        std::pair<float, float>{500.0f, 2000.0f},
        std::pair<float, float>{2000.0f, 6000.0f},
        std::pair<float, float>{6000.0f, 20000.0f},
    };

    std::vector<FeatureFrame> frames(frameCount);
    std::vector<float> previousMagnitudes(spectrumBins, 0.0f);
    std::vector<std::complex<float>> spectrum(windowSize);

    for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex)
    {
        const int frameStart = frameIndex * hopSize;
        float sumSquares = 0.0f;

        for (int i = 0; i < windowSize; ++i)
        {
            const int sampleIndex = frameStart + i;
            const float sample = sampleIndex < sampleCount ? audio.samples[sampleIndex] : 0.0f;
            sumSquares += sample * sample;
            spectrum[i] = std::complex<float>(sample * hann[i], 0.0f);
        }

        frames[frameIndex].rms = std::sqrt(sumSquares / static_cast<float>(windowSize));
        FFT(spectrum);

        std::vector<float> magnitudes(spectrumBins, 0.0f);
        float weightedFreq = 0.0f;
        float magnitudeSum = 0.0f;
        float onset = 0.0f;

        for (int bin = 0; bin < spectrumBins; ++bin)
        {
            const float magnitude = std::abs(spectrum[bin]);
            magnitudes[bin] = magnitude;
            const float frequency = static_cast<float>(bin) * binHz;
            const float power = magnitude * magnitude;
            for (int bandIndex = 0; bandIndex < kBandCount; ++bandIndex)
            {
                const float low = bands[bandIndex].first;
                const float high = std::min(bands[bandIndex].second, nyquist);
                if (frequency >= low && frequency < high)
                {
                    frames[frameIndex].bands[bandIndex] += power;
                }
            }

            weightedFreq += frequency * magnitude;
            magnitudeSum += magnitude;
            if (frameIndex > 0)
            {
                onset += std::max(0.0f, magnitude - previousMagnitudes[bin]);
            }
        }

        for (float& bandEnergy : frames[frameIndex].bands)
        {
            bandEnergy = std::sqrt(std::max(0.0f, bandEnergy));
        }

        frames[frameIndex].centroid = magnitudeSum > 0.0f ? (weightedFreq / magnitudeSum) / std::max(nyquist, 1.0f) : 0.0f;
        frames[frameIndex].onset = onset / static_cast<float>(spectrumBins);
        previousMagnitudes = std::move(magnitudes);
    }

    constexpr int kFeatureCount = 9;
    std::array<std::vector<float>, kFeatureCount> streams;
    for (auto& stream : streams)
    {
        stream.resize(frames.size(), 0.0f);
    }

    for (std::size_t index = 0; index < frames.size(); ++index)
    {
        streams[0][index] = frames[index].rms;
        for (int band = 0; band < kBandCount; ++band)
        {
            streams[1 + band][index] = frames[index].bands[band];
        }
        streams[7][index] = frames[index].centroid;
        streams[8][index] = std::min(frames[index].onset, 10.0f);
    }

    for (auto& stream : streams)
    {
        stream = MovingAverage(stream, 2);
        auto [minIt, maxIt] = std::minmax_element(stream.begin(), stream.end());
        const float minValue = *minIt;
        const float maxValue = *maxIt;
        const float range = maxValue - minValue;
        if (range > 1.0e-6f)
        {
            for (float& value : stream)
            {
                value = Clamp01((value - minValue) / range);
            }
        }
        else
        {
            std::fill(stream.begin(), stream.end(), 0.0f);
        }
    }

    streams[8] = MovingAverage(streams[8], 3);
    for (std::size_t index = 0; index < frames.size(); ++index)
    {
        frames[index].rms = streams[0][index];
        for (int band = 0; band < kBandCount; ++band)
        {
            frames[index].bands[band] = streams[1 + band][index];
        }
        frames[index].centroid = streams[7][index];
        frames[index].onset = Clamp01(streams[8][index]);
    }

    return frames;
}

std::vector<ColumnFeatures> MapFeaturesToColumns(const std::vector<FeatureFrame>& frames, int width)
{
    std::vector<ColumnFeatures> columns(width);
    const int frameCount = static_cast<int>(frames.size());

    for (int x = 0; x < width; ++x)
    {
        const int begin = (x * frameCount) / width;
        const int end = std::max(begin + 1, ((x + 1) * frameCount) / width);
        const int clampedEnd = std::min(end, frameCount);
        const int span = std::max(1, clampedEnd - begin);

        ColumnFeatures aggregated{};
        for (int frameIndex = begin; frameIndex < clampedEnd; ++frameIndex)
        {
            aggregated.rms += frames[frameIndex].rms;
            for (int band = 0; band < kBandCount; ++band)
            {
                aggregated.bands[band] += frames[frameIndex].bands[band];
            }
            aggregated.centroid += frames[frameIndex].centroid;
            aggregated.onset += frames[frameIndex].onset;
        }

        const float invSpan = 1.0f / static_cast<float>(span);
        aggregated.rms *= invSpan;
        for (float& band : aggregated.bands)
        {
            band *= invSpan;
        }
        aggregated.centroid *= invSpan;
        aggregated.onset *= invSpan;
        columns[x] = aggregated;
    }

    return columns;
}

ColumnFeatures AverageColumns(const std::vector<ColumnFeatures>& columns, int begin, int end)
{
    ColumnFeatures result{};
    const int count = std::max(1, end - begin);
    for (int index = begin; index < end; ++index)
    {
        result.rms += columns[index].rms;
        for (int band = 0; band < kBandCount; ++band)
        {
            result.bands[band] += columns[index].bands[band];
        }
        result.centroid += columns[index].centroid;
        result.onset += columns[index].onset;
    }

    const float invCount = 1.0f / static_cast<float>(count);
    result.rms *= invCount;
    for (float& band : result.bands)
    {
        band *= invCount;
    }
    result.centroid *= invCount;
    result.onset *= invCount;
    return result;
}

std::vector<ColumnFeatures> SmoothColumns(const std::vector<ColumnFeatures>& input, int radius, int passes)
{
    if (input.empty() || radius <= 0 || passes <= 0)
    {
        return input;
    }

    std::vector<ColumnFeatures> current = input;
    std::vector<ColumnFeatures> next(input.size());

    for (int pass = 0; pass < passes; ++pass)
    {
        for (int x = 0; x < static_cast<int>(current.size()); ++x)
        {
            const int begin = std::max(0, x - radius);
            const int end = std::min(static_cast<int>(current.size()), x + radius + 1);
            next[x] = AverageColumns(current, begin, end);
        }
        current.swap(next);
    }

    return current;
}

bool WriteImage8(const std::string& path, int width, int height, int comp, const std::vector<std::uint8_t>& bytes)
{
    return stbi_write_png(path.c_str(), width, height, comp, bytes.data(), width * comp) != 0;
}

std::string BuildOutputPath(const std::filesystem::path& outputDir, const char* fileName)
{
    return (outputDir / fileName).string();
}

std::uint32_t Crc32(const std::uint8_t* data, std::size_t length)
{
    std::uint32_t crc = 0xffffffffu;
    for (std::size_t i = 0; i < length; ++i)
    {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
        {
            const std::uint32_t mask = (crc & 1u) ? 0xedb88320u : 0u;
            crc = (crc >> 1) ^ mask;
        }
    }
    return ~crc;
}

std::uint32_t Adler32(const std::vector<std::uint8_t>& data)
{
    std::uint32_t a = 1u;
    std::uint32_t b = 0u;
    for (std::uint8_t byte : data)
    {
        a = (a + byte) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

void AppendBe32(std::vector<std::uint8_t>& bytes, std::uint32_t value)
{
    bytes.push_back(static_cast<std::uint8_t>((value >> 24) & 0xffu));
    bytes.push_back(static_cast<std::uint8_t>((value >> 16) & 0xffu));
    bytes.push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
    bytes.push_back(static_cast<std::uint8_t>(value & 0xffu));
}

void AppendChunk(std::vector<std::uint8_t>& png, const char* tag, const std::vector<std::uint8_t>& payload)
{
    AppendBe32(png, static_cast<std::uint32_t>(payload.size()));
    const std::size_t tagOffset = png.size();
    png.insert(png.end(), tag, tag + 4);
    png.insert(png.end(), payload.begin(), payload.end());
    const std::uint32_t crc = Crc32(png.data() + tagOffset, 4 + payload.size());
    AppendBe32(png, crc);
}

bool WriteGray16Png(const std::string& path, int width, int height, const std::vector<std::uint16_t>& pixels)
{
    if (static_cast<int>(pixels.size()) != width * height)
    {
        return false;
    }

    std::vector<std::uint8_t> raw;
    raw.reserve(static_cast<std::size_t>(height) * (1 + static_cast<std::size_t>(width) * 2u));
    for (int y = 0; y < height; ++y)
    {
        raw.push_back(0u);
        for (int x = 0; x < width; ++x)
        {
            const std::uint16_t value = pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)];
            raw.push_back(static_cast<std::uint8_t>((value >> 8) & 0xffu));
            raw.push_back(static_cast<std::uint8_t>(value & 0xffu));
        }
    }

    std::vector<std::uint8_t> idat;
    idat.reserve(raw.size() + raw.size() / 65535u * 5u + 16u);
    idat.push_back(0x78u);
    idat.push_back(0x01u);

    std::size_t offset = 0;
    while (offset < raw.size())
    {
        const std::size_t remaining = raw.size() - offset;
        const std::uint16_t chunk = static_cast<std::uint16_t>(std::min<std::size_t>(65535u, remaining));
        const bool finalChunk = (offset + chunk) == raw.size();
        idat.push_back(finalChunk ? 0x01u : 0x00u);
        idat.push_back(static_cast<std::uint8_t>(chunk & 0xffu));
        idat.push_back(static_cast<std::uint8_t>((chunk >> 8) & 0xffu));
        const std::uint16_t nlen = static_cast<std::uint16_t>(~chunk);
        idat.push_back(static_cast<std::uint8_t>(nlen & 0xffu));
        idat.push_back(static_cast<std::uint8_t>((nlen >> 8) & 0xffu));
        idat.insert(idat.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
            raw.begin() + static_cast<std::ptrdiff_t>(offset + chunk));
        offset += chunk;
    }

    AppendBe32(idat, Adler32(raw));

    std::vector<std::uint8_t> png = {
        0x89u, 0x50u, 0x4eu, 0x47u, 0x0du, 0x0au, 0x1au, 0x0au,
    };

    std::vector<std::uint8_t> ihdr;
    ihdr.reserve(13);
    AppendBe32(ihdr, static_cast<std::uint32_t>(width));
    AppendBe32(ihdr, static_cast<std::uint32_t>(height));
    ihdr.push_back(16u);
    ihdr.push_back(0u);
    ihdr.push_back(0u);
    ihdr.push_back(0u);
    ihdr.push_back(0u);
    AppendChunk(png, "IHDR", ihdr);
    AppendChunk(png, "IDAT", idat);
    AppendChunk(png, "IEND", {});

    std::ofstream output(path, std::ios::binary);
    if (!output)
    {
        return false;
    }
    output.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    return output.good();
}

bool WriteFeaturesImage(const std::string& path, const std::vector<ColumnFeatures>& columns)
{
    constexpr int rowHeight = 48;
    const int width = static_cast<int>(columns.size());
    const int height = kBandCount * rowHeight;
    std::vector<std::uint8_t> image(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u, 12u);

    const std::array<std::array<std::uint8_t, 3>, kBandCount> colors = {{
        {40u, 70u, 110u},
        {60u, 120u, 70u},
        {160u, 120u, 40u},
        {180u, 80u, 40u},
        {180u, 50u, 80u},
        {210u, 210u, 210u},
    }};

    for (int band = 0; band < kBandCount; ++band)
    {
        const int rowTop = band * rowHeight;
        const int rowBottom = rowTop + rowHeight - 1;
        const int baseLine = rowBottom - 4;

        for (int y = rowTop; y <= rowBottom; ++y)
        {
            const std::uint8_t guide = (y == rowTop || y == rowBottom || y == baseLine) ? 36u : 12u;
            for (int x = 0; x < width; ++x)
            {
                const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 3u;
                image[pixel + 0] = guide;
                image[pixel + 1] = guide;
                image[pixel + 2] = guide;
            }
        }

        for (int x = 0; x < width; ++x)
        {
            const float value = Clamp01(columns[x].bands[band]);
            const int filled = std::max(1, static_cast<int>(std::round(value * static_cast<float>(rowHeight - 8))));
            const std::uint8_t level = static_cast<std::uint8_t>(std::round(70.0f + value * 185.0f));
            for (int y = baseLine; y > baseLine - filled; --y)
            {
                const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 3u;
                image[pixel + 0] = static_cast<std::uint8_t>((static_cast<int>(colors[band][0]) * level) / 255);
                image[pixel + 1] = static_cast<std::uint8_t>((static_cast<int>(colors[band][1]) * level) / 255);
                image[pixel + 2] = static_cast<std::uint8_t>((static_cast<int>(colors[band][2]) * level) / 255);
            }
        }
    }

    return WriteImage8(path, width, height, 3, image);
}

std::vector<float> GenerateHeightField(const std::vector<ColumnFeatures>& columns, const Config& config)
{
    const int size = config.size;
    std::vector<float> heights(static_cast<std::size_t>(size) * static_cast<std::size_t>(size), 0.0f);

    for (int y = 0; y < size; ++y)
    {
        const float ny = static_cast<float>(y) / static_cast<float>(size - 1);
        for (int x = 0; x < size; ++x)
        {
            const float nx = static_cast<float>(x) / static_cast<float>(size - 1);
            const ColumnFeatures& feature = columns[x];
            const float low = 0.65f * feature.bands[0] + 0.35f * feature.bands[1];
            const float mid = 0.6f * feature.bands[2] + 0.4f * feature.bands[3];
            const float high = 0.4f * feature.bands[4] + 0.6f * feature.bands[5];

            const float base = Fbm(nx * 4.0f, ny * 4.0f, 5, 2.0f, 0.5f, config.seed);
            const float ridgeFreq = 2.5f + 10.0f * low;
            const float ridgeNoise = ValueNoise(nx * ridgeFreq, ny * (1.8f + 4.0f * low), config.seed + 17u);
            const float ridge = 1.0f - std::fabs(2.0f * ridgeNoise - 1.0f);
            const float macroWarp = Fbm(nx * (1.5f + 3.0f * mid), ny * 1.6f, 3, 2.2f, 0.55f, config.seed + 41u);
            const float detail = Fbm(nx * (18.0f + 16.0f * high), ny * (18.0f + 16.0f * high), 3, 2.0f, 0.55f, config.seed + 67u);
            const float slope = (feature.centroid - 0.5f) * (ny - 0.5f) * 0.25f;

            float height = 0.28f + base * 0.52f;
            height += ridge * (0.08f + 0.34f * low);
            height += (macroWarp - 0.5f) * (0.08f + 0.12f * mid);
            height += (detail - 0.5f) * (0.04f + 0.14f * high);
            height += slope;

            heights[static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x)] = height;
        }
    }

    return heights;
}

void ApplyThermalErosion(std::vector<float>& heights, const std::vector<ColumnFeatures>& columns, int size)
{
    std::vector<float> delta(heights.size(), 0.0f);
    int maxIterations = 0;
    std::vector<int> columnIterations(size, 0);

    for (int x = 0; x < size; ++x)
    {
        const float drive = 0.65f * columns[x].bands[3] + 0.35f * columns[x].onset;
        columnIterations[x] = 1 + static_cast<int>(std::round(drive * 5.0f));
        maxIterations = std::max(maxIterations, columnIterations[x]);
    }

    constexpr std::array<int, 4> dx = {1, -1, 0, 0};
    constexpr std::array<int, 4> dy = {0, 0, 1, -1};

    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        std::fill(delta.begin(), delta.end(), 0.0f);
        for (int y = 1; y < size - 1; ++y)
        {
            for (int x = 1; x < size - 1; ++x)
            {
                if (iteration >= columnIterations[x])
                {
                    continue;
                }

                const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x);
                const float talus = 0.012f + 0.02f * (1.0f - columns[x].bands[3]);
                float totalTransfer = 0.0f;

                for (int neighbor = 0; neighbor < 4; ++neighbor)
                {
                    const int nx = x + dx[neighbor];
                    const int ny = y + dy[neighbor];
                    const std::size_t neighborIndex = static_cast<std::size_t>(ny) * static_cast<std::size_t>(size) + static_cast<std::size_t>(nx);
                    const float diff = heights[index] - heights[neighborIndex];
                    if (diff > talus)
                    {
                        const float transfer = (diff - talus) * 0.125f;
                        totalTransfer += transfer;
                        delta[neighborIndex] += transfer;
                    }
                }

                delta[index] -= totalTransfer;
            }
        }

        for (std::size_t index = 0; index < heights.size(); ++index)
        {
            heights[index] += delta[index];
        }
    }
}

void ApplyRainCarve(std::vector<float>& heights, const std::vector<ColumnFeatures>& columns, int size)
{
    for (int pass = 0; pass < 2; ++pass)
    {
        std::vector<float> next = heights;
        for (int y = 1; y < size - 1; ++y)
        {
            for (int x = 1; x < size - 1; ++x)
            {
                const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x);
                const float rainfall = 0.002f + columns[x].rms * 0.012f;

                const std::size_t left = index - 1;
                const std::size_t right = index + 1;
                const std::size_t up = index - static_cast<std::size_t>(size);
                const std::size_t down = index + static_cast<std::size_t>(size);

                const float lowest = std::min(std::min(heights[left], heights[right]), std::min(heights[up], heights[down]));
                const float slope = heights[index] - lowest;
                if (slope <= 0.0f)
                {
                    continue;
                }

                const float carve = std::min(slope * 0.2f, rainfall);
                next[index] -= carve;
                if (lowest == heights[left])
                {
                    next[left] += carve * 0.35f;
                }
                else if (lowest == heights[right])
                {
                    next[right] += carve * 0.35f;
                }
                else if (lowest == heights[up])
                {
                    next[up] += carve * 0.35f;
                }
                else
                {
                    next[down] += carve * 0.35f;
                }
            }
        }
        heights.swap(next);
    }
}

void NormalizeHeights(std::vector<float>& heights, float seaLevel)
{
    auto [minIt, maxIt] = std::minmax_element(heights.begin(), heights.end());
    const float minValue = *minIt;
    const float maxValue = *maxIt;
    const float range = std::max(1.0e-6f, maxValue - minValue);

    for (float& height : heights)
    {
        height = Clamp01((height - minValue) / range);
        height = Clamp01((height - seaLevel) / std::max(1.0f - seaLevel, 1.0e-6f));
    }
}

std::vector<std::uint16_t> QuantizeHeight16(const std::vector<float>& heights)
{
    std::vector<std::uint16_t> result(heights.size(), 0u);
    for (std::size_t i = 0; i < heights.size(); ++i)
    {
        result[i] = static_cast<std::uint16_t>(std::round(Clamp01(heights[i]) * 65535.0f));
    }
    return result;
}

std::vector<std::uint8_t> BuildNormalMap(const std::vector<float>& heights, int size)
{
    std::vector<std::uint8_t> normal(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 3u, 0u);
    const float scale = 24.0f / static_cast<float>(size);

    for (int y = 0; y < size; ++y)
    {
        const int y0 = std::max(0, y - 1);
        const int y1 = std::min(size - 1, y + 1);
        for (int x = 0; x < size; ++x)
        {
            const int x0 = std::max(0, x - 1);
            const int x1 = std::min(size - 1, x + 1);

            const float hx0 = heights[static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x0)];
            const float hx1 = heights[static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x1)];
            const float hy0 = heights[static_cast<std::size_t>(y0) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x)];
            const float hy1 = heights[static_cast<std::size_t>(y1) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x)];

            float nx = -(hx1 - hx0) * scale;
            float ny = -(hy1 - hy0) * scale;
            float nz = 1.0f;
            const float invLength = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            nx *= invLength;
            ny *= invLength;
            nz *= invLength;

            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x)) * 3u;
            normal[pixel + 0] = static_cast<std::uint8_t>(std::round((nx * 0.5f + 0.5f) * 255.0f));
            normal[pixel + 1] = static_cast<std::uint8_t>(std::round((ny * 0.5f + 0.5f) * 255.0f));
            normal[pixel + 2] = static_cast<std::uint8_t>(std::round((nz * 0.5f + 0.5f) * 255.0f));
        }
    }

    return normal;
}

std::vector<std::uint8_t> BuildPreview(const std::vector<float>& heights, int size)
{
    std::vector<std::uint8_t> preview(static_cast<std::size_t>(size) * static_cast<std::size_t>(size) * 3u, 0u);
    const std::array<float, 3> light = {0.45f, -0.35f, 0.82f};

    for (int y = 0; y < size; ++y)
    {
        const int y0 = std::max(0, y - 1);
        const int y1 = std::min(size - 1, y + 1);
        for (int x = 0; x < size; ++x)
        {
            const int x0 = std::max(0, x - 1);
            const int x1 = std::min(size - 1, x + 1);

            const float h = heights[static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x)];
            const float hx0 = heights[static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x0)];
            const float hx1 = heights[static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x1)];
            const float hy0 = heights[static_cast<std::size_t>(y0) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x)];
            const float hy1 = heights[static_cast<std::size_t>(y1) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x)];

            float nx = -(hx1 - hx0) * 18.0f;
            float ny = -(hy1 - hy0) * 18.0f;
            float nz = 1.0f;
            const float invLength = 1.0f / std::sqrt(nx * nx + ny * ny + nz * nz);
            nx *= invLength;
            ny *= invLength;
            nz *= invLength;

            const float lambert = std::max(0.0f, nx * light[0] + ny * light[1] + nz * light[2]);
            const float ambient = 0.35f;
            const float shade = ambient + lambert * 0.65f;

            std::array<float, 3> baseColor{};
            if (h < 0.18f)
            {
                baseColor = {0.05f, 0.18f, 0.28f};
            }
            else if (h < 0.42f)
            {
                baseColor = {0.18f, 0.34f, 0.16f};
            }
            else if (h < 0.7f)
            {
                baseColor = {0.44f, 0.34f, 0.22f};
            }
            else
            {
                baseColor = {0.72f, 0.72f, 0.74f};
            }

            const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(size) + static_cast<std::size_t>(x)) * 3u;
            preview[pixel + 0] = static_cast<std::uint8_t>(std::round(Clamp01(baseColor[0] * shade) * 255.0f));
            preview[pixel + 1] = static_cast<std::uint8_t>(std::round(Clamp01(baseColor[1] * shade) * 255.0f));
            preview[pixel + 2] = static_cast<std::uint8_t>(std::round(Clamp01(baseColor[2] * shade) * 255.0f));
        }
    }

    return preview;
}
} // namespace

int main(int argc, char** argv)
{
    Config config;
    if (!ParseArgs(argc, argv, config))
    {
        PrintUsage();
        return 1;
    }

    AudioData audio;
    std::string error;
    if (!LoadWav(config.inputPath, audio, error))
    {
        std::cerr << "WAV load failed: " << error << "\n";
        return 1;
    }

    const std::filesystem::path outputDir = "out";
    std::error_code fsError;
    std::filesystem::create_directories(outputDir, fsError);
    if (fsError)
    {
        std::cerr << "Failed to create output folder: " << outputDir.string() << "\n";
        return 1;
    }

    const std::vector<FeatureFrame> frames = AnalyzeAudio(audio, config);
    const std::vector<ColumnFeatures> mappedColumns = MapFeaturesToColumns(frames, config.size);
    const std::vector<ColumnFeatures> columns = SmoothColumns(mappedColumns, 4, 2);
    const std::string featuresPath = BuildOutputPath(outputDir, "features.png");
    const std::string heightPath = BuildOutputPath(outputDir, "height.png");
    const std::string normalPath = BuildOutputPath(outputDir, "normal.png");
    const std::string previewPath = BuildOutputPath(outputDir, "preview.png");
    const std::string debugPath = BuildOutputPath(outputDir, "debug_features.png");

    if (!WriteFeaturesImage(featuresPath, columns))
    {
        std::cerr << "Failed to write " << featuresPath << "\n";
        return 1;
    }

    std::vector<float> heights = GenerateHeightField(columns, config);
    if (!config.disableErosion)
    {
        ApplyThermalErosion(heights, columns, config.size);
        ApplyRainCarve(heights, columns, config.size);
    }
    NormalizeHeights(heights, 0.08f);

    const std::vector<std::uint16_t> height16 = QuantizeHeight16(heights);
    if (!WriteGray16Png(heightPath, config.size, config.size, height16))
    {
        std::cerr << "Failed to write " << heightPath << "\n";
        return 1;
    }

    const std::vector<std::uint8_t> normal = BuildNormalMap(heights, config.size);
    if (!WriteImage8(normalPath, config.size, config.size, 3, normal))
    {
        std::cerr << "Failed to write " << normalPath << "\n";
        return 1;
    }

    const std::vector<std::uint8_t> preview = BuildPreview(heights, config.size);
    if (!WriteImage8(previewPath, config.size, config.size, 3, preview))
    {
        std::cerr << "Failed to write " << previewPath << "\n";
        return 1;
    }

    if (config.exportDebug)
    {
        const int width = config.size;
        constexpr int rowHeight = 28;
        const int height = 9 * rowHeight;
        std::vector<std::uint8_t> debug(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3u, 10u);
        for (int x = 0; x < width; ++x)
        {
            const std::array<float, 9> values = {
                columns[x].rms,
                columns[x].bands[0],
                columns[x].bands[1],
                columns[x].bands[2],
                columns[x].bands[3],
                columns[x].bands[4],
                columns[x].bands[5],
                columns[x].centroid,
                columns[x].onset,
            };

            for (int row = 0; row < 9; ++row)
            {
                const int rowTop = row * rowHeight;
                const int rowBottom = rowTop + rowHeight - 1;
                const int baseLine = rowBottom - 3;
                const std::uint8_t line = 28u;
                for (int y = rowTop; y <= rowBottom; ++y)
                {
                    if (y != rowTop && y != rowBottom && y != baseLine)
                    {
                        continue;
                    }
                    const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 3u;
                    debug[pixel + 0] = line;
                    debug[pixel + 1] = line;
                    debug[pixel + 2] = line;
                }

                const int filled = std::max(1, static_cast<int>(std::round(Clamp01(values[row]) * static_cast<float>(rowHeight - 6))));
                const std::uint8_t level = static_cast<std::uint8_t>(std::round(70.0f + Clamp01(values[row]) * 185.0f));
                for (int y = baseLine; y > baseLine - filled; --y)
                {
                    const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x)) * 3u;
                    debug[pixel + 0] = level;
                    debug[pixel + 1] = level;
                    debug[pixel + 2] = level;
                }
            }
        }

        if (!WriteImage8(debugPath, width, height, 3, debug))
        {
            std::cerr << "Failed to write " << debugPath << "\n";
            return 1;
        }
    }

    std::cout
        << "Generated outputs in " << outputDir.string() << "\n"
        << "  " << heightPath << "\n"
        << "  " << normalPath << "\n"
        << "  " << featuresPath << "\n"
        << "  " << previewPath << "\n"
        << "Sample rate: " << audio.sampleRate << " Hz\n"
        << "Frames: " << frames.size() << "\n"
        << "Terrain size: " << config.size << "x" << config.size << "\n";
    return 0;
}
