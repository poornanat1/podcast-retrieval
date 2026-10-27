# Voice robustness: what transcription costs retrieval

Spoken queries measured end to end: a synthetic speech test set of 1,240
clips, transcribed by Whisper small through the running API, scored
against the same queries typed. Measured 2026-10-26 on a laptop (speech
server on the Apple GPU, Postgres in Docker). The design is in
[docs/15-voice.md](../docs/15-voice.md); the numbers are in
[speech/results-v1-quality.json](speech/results-v1-quality.json) and
[speech/results-v1-latency.json](speech/results-v1-latency.json).

**The test set is synthetic** (Chatterbox speaking each query in cloned
macOS voices, see the design doc), so everything here is a lower bound on
what real users will experience: clean, articulate speech, no
disfluencies, accents that are one model's imitation of another's. The
findings about *what kind* of error hurts retrieval, and by how much, are
what transfers.

## Headline

On the 180 relevance queries spoken clean in the base voice:

| | typed | spoken | loss |
|---|---|---|---|
| word error rate | | 4.2% (84% of clips exact) | |
| named-entity error rate (name queries) | | 23% | |
| NDCG@10 | 0.907 | 0.825 | **0.082** (95% CI 0.057–0.110) |
| MRR | 0.984 | 0.956 | 0.028 |
| Recall@100 | 0.505 | 0.445 | 0.060 |

A 4% word error rate costs 8 points of NDCG@10, because the errors are
not spread evenly: they land on names. The loss splits in two.

1. **Clips with word errors (28 of 180) lose 0.32 NDCG@10 each.** 23 of
   them are name queries, and 13 of those are near misses, a name heard
   as a homophone: *Lewis House* for Lewis Howes, *Joe Roken*, *Matt
   Wall*, *Joel Austin*, *Cara Berry*, *Megan Kelly*, *Ramsay*. A near
   miss loses as much as an omission (0.43 vs 0.48 NDCG@10 across the
   whole set), because lexical search matches whole words and the vector
   legs do not know the catalog's spelling.
2. **Clips with zero word errors (152 of 180) still lose 0.038.** That is
   the recognizer's casing and punctuation. Whisper capitalizes names and
   hyphenates compounds; the vector encoders are case-sensitive, so *Joe
   Rogan* retrieves differently from *joe rogan* (0.24 NDCG@10 lost on
   that query, 0.57 on *the Times*), and *stand-up* no longer matches
   *stand up* lexically (0.60 lost). Lowercasing the transcript before
   searching recovers most of it: on the base tier NDCG@10 goes from
   0.825 to 0.851, and on the zero-error clips from 0.871 to 0.902, the
   typed reference's level. This is a one-line mitigation, and it applies
   to typed queries too: anyone who types *Joe Rogan* today gets the
   degraded result.

By query type, on the base tier:

| type | clips | WER | NEER | typed NDCG@10 | spoken | loss |
|---|---|---|---|---|---|---|
| entity | 20 | 6.3% | 20% | 1.000 | 0.829 | 0.171 |
| navigational | 40 | 5.7% | 25% | 0.992 | 0.882 | 0.110 |
| filtered | 35 | 7.5% | | 0.813 | 0.707 | 0.106 |
| exploratory | 85 | 1.6% | | 0.883 | 0.845 | 0.038 |

Exploratory queries are nearly immune (topic words are common words);
name queries carry the damage.

## Accents

The 60-query subset in six cloned English voices, all clean:

| voice | WER | exact clips | NEER | spoken NDCG@10 | loss vs typed (0.921) |
|---|---|---|---|---|---|
| en_AU | 3.0% | 88% | 24% | 0.891 | 0.030 |
| en_GB | 3.6% | 87% | 20% | 0.869 | 0.052 |
| en_US (base) | 3.5% | 85% | 28% | 0.845 | 0.076 |
| en_ZA | 6.8% | 72% | 32% | 0.846 | 0.075 |
| en_IE | 5.6% | 82% | 28% | 0.835 | 0.086 |
| en_IN | 10.9% | 67% | 32% | 0.745 | 0.176 |

The Indian-English voice triples the word error rate and more than
doubles the retrieval loss. Its errors are of two kinds: names again
(*Glenn Back*, *NewsGust* for newscast, *the 5-year sports show*) and the
carrier phrase itself (*Touch for the ringer*), which query understanding
then fails to strip, so the search text keeps *touch for*. Whether this
reflects Indian English or Chatterbox's rendering of one macOS voice
cannot be told from synthetic data; it is the strongest argument for a
small set of human recordings.

## Noise, microphones, rooms

The same 60 queries, base voice, degraded from the clean clip:

| condition | WER | exact clips | NEER | spoken NDCG@10 | loss |
|---|---|---|---|---|---|
| clean | 3.5% | 85% | 28% | 0.845 | 0.076 |
| white noise 20 dB SNR | 5.0% | 83% | 32% | 0.832 | 0.089 |
| white noise 10 dB | 5.7% | 78% | 32% | 0.831 | 0.090 |
| laptop microphone | 5.8% | 80% | 32% | 0.830 | 0.091 |
| background speech 10 dB | 8.0% | 72% | 36% | 0.832 | 0.089 |
| background speech 5 dB | 9.1% | 67% | 24% | 0.821 | 0.100 |
| telephone band | 5.2% | 80% | 28% | 0.811 | 0.109 |
| room reverberation | 6.3% | 75% | 36% | 0.815 | 0.106 |
| white noise 5 dB | 11.5% | 57% | 44% | 0.810 | 0.111 |
| white noise 0 dB | 32.5% | 27% | 64% | 0.532 | 0.389 |

Whisper is robust down to 5 dB: every condition short of speech-level
noise costs 1–4 points over clean. At 0 dB (noise as loud as the voice)
it collapses, and names go first (64% entity errors: *Megantally show*,
*what's happened* for Watch What Crappens).

## Speaking rate

| rate | WER | exact clips | spoken NDCG@10 | loss |
|---|---|---|---|---|
| 0.8× | 6.0% | 77% | 0.822 | 0.099 |
| 1.0× | 3.5% | 85% | 0.845 | 0.076 |
| 1.3× | 6.4% | 78% | 0.826 | 0.095 |
| 1.6× | 9.9% | 65% | 0.786 | 0.135 |

Both slower and faster speech cost about 2 points; very fast speech costs
6. (Rates are phase-vocoder time stretches of the clean clip, pitch
preserved, so this isolates tempo from everything else a fast speaker
changes.)

## Phrasing: what is spoken is not the labeled query

Spoken queries come wrapped: the terse clips in the carrier *"Search for
…"*, and 40 exploratory and filtered queries as requests (*"could you find
me some podcast episodes about …"*). Query understanding strips the
framing. Typing the wrapped text, against typing the labeled query with
understanding off:

| phrasing | labeled query NDCG@10 | wrapped text typed | cost of the phrasing | then spoken | cost of transcription |
|---|---|---|---|---|---|
| terse with carrier (180) | 0.936 | 0.907 | 0.030 | 0.825 | 0.082 |
| conversational (40) | 0.881 | 0.801 | 0.080 | 0.757 | 0.044 |

The phrasing costs come from a few queries, and the cause is the same
in both rows: the labeled relevance judgments were made on the full
query text, and understanding removes words from it. For *"family
friendly history storytelling"* it sets the no-explicit filter and
searches *history storytelling* (0.91 NDCG@10 lost); for *"kid friendly
science episodes"* it searches *science episodes* (0.86); for *"science
explainers under 30 minutes"* it drops the duration phrase (0.72). The
filter is applied, but the words also carried lexical evidence the
judges rewarded. This is a property of typed search that the voice
path inherits, and a candidate fix is to keep the filter phrase in the
lexical text while still applying the filter. The conversational
templates themselves (*could you find me*, *are there any podcasts that
talk about*, *please*) are all stripped cleanly, 0.00 cost on 33 of 40.

Conversational clips transcribe better than terse ones (2.9% vs 4.2%
WER): longer context helps the recognizer, and the topic words are
ordinary vocabulary.

## Non-English queries

Six queries are in Spanish, French, German or Portuguese, spoken by the
matching cloned voice with the English carrier. Language detection got
five of six (the Portuguese clip was detected as Albanian and
transcribed as gibberish, losing all of its NDCG@10); the English carrier
in a foreign-language clip confuses detection (*Dutch for Wissenschaft
Podcast*). Six clips is too few for a number; the mechanism, that a
language hint is worth sending when the UI knows it, is clear, and the
API accepts one.

## Confidence

The recognizer's mean token log-probability separates its failures:
−0.18 on clips it got exactly right, −0.34 on clips with errors, and
below −0.4 on most wrong names. It is stored with every voice search,
so a low-confidence transcript can be flagged in the UI, or routed to an
N-best or correction step, without any new signal.

## Latency

Base and conversational tiers (220 clips) through the production
configuration (lexical leg limited to 300 ms), one client, in sequence.
Speech server on the Apple GPU, Whisper small, greedy decoding.

| audio length | clips | transcription p50 / p95 ms | search after transcription p50 / p95 | end to end p50 / p95 |
|---|---|---|---|---|
| under 2 s | 36 | 411 / 965 | 326 / 635 | 774 / 1457 |
| 2–4 s | 166 | 614 / 1037 | 172 / 368 | 772 / 1365 |
| 4–8 s | 18 | 862 / 1288 | 98 / 1107 | 1050 / 1883 |
| **all** | 220 | **710 / 1037** | **189 / 415** | **866 / 1380** |

Transcription dominates and grows with the clip: about 0.4 s for a
two-second query, 0.9 s for a five-second one, 1.0–1.3 s at p95. The
search after it is the text path measured in the serving report (here
189 ms p50, 415 ms p95, with the lexical leg timing out on
69 of 220), so the text budget of 500 ms p95 excluding
transcription holds on this run, and voice adds about a second on top. On
the CPU, under load, the same transcription took 2–4 s; the GPU is not
optional for interactive voice. A distilled or int8 Whisper, or streaming
transcription while the user is still speaking, are the ways down.

## Candidate mitigations, in order of expected value

1. **Normalize the transcript before search: lowercase, and treat
   hyphens as spaces.** Measured: recovers 2.6 of the 8.2 points on the
   base tier and nearly all of the loss on error-free clips. Apply it to
   typed queries as well.
2. **Catalog-aware entity correction.** Near misses are the largest
   remaining loss (116 near-miss clips at 0.43 NDCG@10 each). The
   transcript's n-grams can be matched against show, host and publisher
   names by phonetic key (Double Metaphone) or character edit distance
   under a threshold, since *Lewis House → Lewis Howes* is a 1-character
   edit and *Joel Austin → Joel Osteen* a phonetic match. Gate it on the
   recognizer's confidence to avoid "correcting" real queries.
3. **Keep filter phrases in the lexical text.** Fixes the phrasing cost
   on filtered queries for voice and typed search alike.
4. **Send the language hint** when the UI language filter is set, and
   detect the language on the clip with the carrier removed.
5. **N-best transcripts** fused by RRF, so a name's second-best spelling
   gets a vote. Not measured; Whisper's beam search exposes it.
6. **Train with realistically corrupted queries**: the error distribution
   here (homophone substitutions on names, casing) is simple to simulate
   for the two-tower model's training queries.

## What this does not show

- **Synthetic speech, synthetic accents.** No human recordings; the
  numbers are a lower bound and the accent ranking is not evidence about
  real speakers. Twenty queries read by a handful of people would
  calibrate every table above.
- **In-sample absolute metrics.** The served ranker was trained on these
  queries; only the typed-versus-spoken differences are meaningful.
- **One recognizer.** Whisper small, greedy decoding. Larger Whisper
  models or a hosted recognizer will have different, probably smaller,
  name error rates; the harness takes either through `PODFIND_STT`.
- **Named entities are the name queries only.** Exploratory queries with
  a name inside them (*mountaineering disasters on everest*) are not
  annotated, so NEER understates entity errors overall.
- **Twenty lexical time-outs** out of 1,240 clips in the quality run, at a
  25 s limit on a cold cache, counted against the spoken side.
