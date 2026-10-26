package search

import (
	"encoding/binary"
	"errors"
	"testing"
)

// testWAV builds a 16-bit mono PCM WAV of silence.
func testWAV(rate int, seconds float64) []byte {
	n := int(float64(rate) * seconds)
	b := make([]byte, 44+2*n)
	copy(b[0:], "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(36+2*n))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], wavPCM)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], uint32(rate))
	binary.LittleEndian.PutUint32(b[28:], uint32(rate*2))
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(2*n))
	return b
}

func TestParseWAV(t *testing.T) {
	info, err := ParseWAV(testWAV(16000, 2.5))
	if err != nil {
		t.Fatal(err)
	}
	if info.Channels != 1 || info.SampleRate != 16000 || info.Bits != 16 || info.Seconds != 2.5 || info.Float {
		t.Fatalf("info %+v", info)
	}
	for name, data := range map[string][]byte{
		"not wav":   []byte("ID3 definitely not audio"),
		"too short": testWAV(16000, 0.05),
		"too long":  testWAV(8000, MaxVoiceSeconds+1),
		"truncated": testWAV(16000, 1)[:30],
	} {
		if _, err := ParseWAV(data); !errors.Is(err, ErrBadRequest) {
			t.Errorf("%s: err %v", name, err)
		}
	}
	// A 32-bit float header is accepted; a 12-bit one is not.
	f := testWAV(24000, 1)
	binary.LittleEndian.PutUint16(f[20:], wavFloat)
	binary.LittleEndian.PutUint16(f[34:], 32)
	if info, err := ParseWAV(f); err != nil || !info.Float {
		t.Fatalf("float: %+v %v", info, err)
	}
	binary.LittleEndian.PutUint16(f[34:], 12)
	if _, err := ParseWAV(f); !errors.Is(err, ErrBadRequest) {
		t.Fatalf("12-bit: %v", err)
	}
}
