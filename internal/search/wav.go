package search

import (
	"encoding/binary"
	"fmt"
)

// WAVInfo describes an uploaded recording.
type WAVInfo struct {
	Channels   int
	SampleRate int
	Bits       int
	Float      bool
	Seconds    float64
}

const (
	wavPCM        = 1
	wavFloat      = 3
	wavExtensible = 0xFFFE
	// Voice queries are short; the speech server rejects anything over 60 s
	// and the API stops well before that.
	MaxVoiceSeconds = 30.0
	MaxVoiceBytes   = 4 << 20
)

// ParseWAV validates a RIFF/WAVE header (PCM 8/16/24/32-bit or 32-bit
// float, any rate from 8 kHz) without decoding the samples; the speech
// server does the decoding.
func ParseWAV(data []byte) (WAVInfo, error) {
	if len(data) < 12 || string(data[:4]) != "RIFF" || string(data[8:12]) != "WAVE" {
		return WAVInfo{}, fmt.Errorf("%w: audio must be a WAV file", ErrBadRequest)
	}
	var fmtChunk []byte
	dataLen := -1
	for pos := 12; pos+8 <= len(data); {
		id := string(data[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		body := data[pos+8:]
		if size > len(body) {
			size = len(body)
		}
		switch id {
		case "fmt ":
			fmtChunk = body[:size]
		case "data":
			dataLen = size
		}
		if dataLen >= 0 {
			break
		}
		pos += 8 + size + size&1
	}
	if len(fmtChunk) < 16 || dataLen < 0 {
		return WAVInfo{}, fmt.Errorf("%w: WAV file has no fmt or data chunk", ErrBadRequest)
	}
	tag := int(binary.LittleEndian.Uint16(fmtChunk[0:2]))
	if tag == wavExtensible && len(fmtChunk) >= 26 {
		tag = int(binary.LittleEndian.Uint16(fmtChunk[24:26]))
	}
	info := WAVInfo{
		Channels:   int(binary.LittleEndian.Uint16(fmtChunk[2:4])),
		SampleRate: int(binary.LittleEndian.Uint32(fmtChunk[4:8])),
		Bits:       int(binary.LittleEndian.Uint16(fmtChunk[14:16])),
		Float:      tag == wavFloat,
	}
	switch {
	case tag == wavPCM && (info.Bits == 8 || info.Bits == 16 || info.Bits == 24 || info.Bits == 32):
	case tag == wavFloat && info.Bits == 32:
	default:
		return WAVInfo{}, fmt.Errorf("%w: unsupported WAV encoding (format %d, %d bits)", ErrBadRequest, tag, info.Bits)
	}
	if info.Channels < 1 || info.Channels > 2 || info.SampleRate < 8000 || info.SampleRate > 96000 {
		return WAVInfo{}, fmt.Errorf("%w: unsupported WAV layout (%d channels, %d Hz)", ErrBadRequest, info.Channels, info.SampleRate)
	}
	frame := info.Channels * info.Bits / 8
	info.Seconds = float64(dataLen/frame) / float64(info.SampleRate)
	if info.Seconds < 0.1 {
		return WAVInfo{}, fmt.Errorf("%w: recording is shorter than 0.1 s", ErrBadRequest)
	}
	if info.Seconds > MaxVoiceSeconds {
		return WAVInfo{}, fmt.Errorf("%w: recording is %.1f s; at most %.0f s is accepted", ErrBadRequest, info.Seconds, MaxVoiceSeconds)
	}
	return info, nil
}
