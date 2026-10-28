# Voice robustness: what transcription costs retrieval

Spoken queries measured end to end: a synthetic speech test set of 1,240
clips, transcribed by Whisper small through the running API, scored
against the same queries typed. Measured 2026-10-28 on a laptop (speech
server on the Apple GPU, Postgres in Docker). The design is in
[docs/15-voice.md](../docs/15-voice.md); the numbers are in
[speech/results-v2-quality.json](speech/results-v2-quality.json) and
[speech/results-v2-latency.json](speech/results-v2-latency.json).

The reported set is **v2**, synthesized by ElevenLabs Eleven v4 with
library voices chosen by accent and language. A first set, v1, used
Chatterbox cloning macOS voices; it is kept, and the last section says
what changed between them and why v1 was dropped. **Both are synthetic.**
Everything here is a lower bound on what real users will experience:
clean, articulate speech, no disfluencies. The findings about *what kind*
of error hurts retrieval, and by how much, are what transfers.

## Headline

On the 180 relevance queries spoken bare in the base voice (US English):

| | typed | spoken | loss |
|---|---|---|---|
| word error rate | | 5.3% (90% of clips exact) | |
| named-entity error rate (name queries) | | 22% | |
| NDCG@10 | 0.906 | 0.820 | **0.087** (95% CI 0.064–0.113) |
| MRR | 0.984 | 0.951 | 0.033 |
| Recall@100 | 0.505 | 0.431 | 0.074 |

Nine clips in ten transcribe perfectly, and retrieval still loses 9 points
of NDCG@10. The loss has two sources of about equal size.

1. **Clips with word errors (18 of 180) lose 0.28 NDCG@10 each**, and
   nearly all of them are names heard as homophones: *Lewis House* for
   Lewis Howes, *Matt Wall*, *Megan Kelly*, *Ramsay*, *Carroberry* for
   Kara Berry, *of funny* for Kinda Funny. Across the whole set a near
   miss loses 0.38 NDCG@10 and an omission 0.63, because lexical search
   matches whole words and the vector legs do not know the catalog's
   spelling. These errors are speaker-independent: the same names fail in
   every voice, in v1 and v2 alike.
2. **Clips with zero word errors (162 of 180) lose 0.066.** That is the
   recognizer's casing and punctuation. Whisper writes a bare query as a
   title, *Formula One Race Strategy.*, *The Times*, *How Casinos Make
   Money*, and the vector encoders are case-sensitive: *The Times* loses
   0.72 NDCG@10 against *the times*, *The Joe Rogan Experience* 0.43.
   Lowercasing the transcript before searching recovers most of it: on
   the base tier NDCG@10 goes from 0.820 to 0.877, two thirds of the
   whole loss, and on the error-free clips from 0.832 to 0.895, within a
   point of typed. It is a one-line mitigation, and it applies to typed
   queries too: anyone who types *Joe Rogan* today gets the degraded
   result.

By query type, on the base tier:

| type | clips | WER | NEER | typed NDCG@10 | spoken | loss |
|---|---|---|---|---|---|---|
| entity | 20 | 15.0% | 20% | 1.000 | 0.804 | 0.197 |
| navigational | 40 | 8.4% | 23% | 0.992 | 0.919 | 0.073 |
| exploratory | 85 | 2.5% | | 0.882 | 0.800 | 0.082 |
| filtered | 35 | 2.9% | | 0.813 | 0.763 | 0.050 |

Name queries carry the recognition errors; exploratory queries transcribe
almost perfectly and lose to casing.

## Accents

The 60-query subset in six library voices, all clean. These are voices of
real speakers (cloned by ElevenLabs), so the accents are genuine.

| voice | WER | exact clips | NEER | spoken NDCG@10 | loss vs typed (0.921) |
|---|---|---|---|---|---|
| en_AU | 7.8% | 87% | 24% | 0.837 | 0.084 |
| en_GB | 6.7% | 87% | 28% | 0.836 | 0.084 |
| en_US (base) | 9.0% | 85% | 28% | 0.830 | 0.091 |
| en_IN | 10.6% | 82% | 32% | 0.827 | 0.094 |
| en_IE | 7.5% | 88% | 24% | 0.826 | 0.095 |
| en_ZA | 8.1% | 87% | 28% | 0.817 | 0.104 |

**Accent barely matters** for this recognizer on clear speech: the whole
spread is two points of NDCG@10, and the ordering is within noise for 60
queries. The Indian-English voice has the highest word error rate, but
its errors are the same names everyone else gets wrong (*Lewis House*,
*Megan Kelly*) plus a few more (*The Rinner*, *Jerry Mandarin explained*).
This reverses v1's finding, where the cloned Indian voice tripled the
error rate; see the last section.

## Noise, microphones, rooms

The same 60 queries, base voice, degraded from the clean clip:

| condition | WER | exact clips | NEER | spoken NDCG@10 | loss |
|---|---|---|---|---|---|
| laptop microphone | 8.3% | 90% | 24% | 0.865 | 0.055 |
| white noise 20 dB SNR | 10.0% | 87% | 28% | 0.848 | 0.073 |
| telephone band | 6.7% | 90% | 24% | 0.837 | 0.084 |
| clean | 9.0% | 85% | 28% | 0.830 | 0.091 |
| background speech 10 dB | 11.8% | 82% | 32% | 0.825 | 0.096 |
| white noise 10 dB | 13.5% | 80% | 36% | 0.822 | 0.099 |
| room reverberation | 8.9% | 85% | 24% | 0.817 | 0.104 |
| white noise 5 dB | 14.4% | 73% | 44% | 0.763 | 0.158 |
| background speech 5 dB | 17.4% | 65% | 44% | 0.763 | 0.158 |
| white noise 0 dB | 34.9% | 40% | 68% | 0.561 | 0.360 |

Down to 10 dB nothing costs more than a point or two over clean, and
three conditions score above clean, which is the noise of 60 queries,
not a benefit. At 5 dB, noise or competing speech, the loss doubles; at
0 dB the recognizer collapses and names go first (68% entity errors).

## Speaking rate

| rate | WER | exact clips | spoken NDCG@10 | loss |
|---|---|---|---|---|
| 0.8× | 9.6% | 85% | 0.823 | 0.098 |
| 1.0× | 9.0% | 85% | 0.830 | 0.091 |
| 1.3× | 9.7% | 78% | 0.816 | 0.105 |
| 1.6× | 14.7% | 75% | 0.762 | 0.159 |

Slower and moderately faster speech cost about a point; very fast speech
costs seven. (Rates are phase-vocoder time stretches of the clean clip,
pitch preserved, so this isolates tempo from everything else a fast
speaker changes.)

## Phrasing: what is spoken is not the labeled query

Forty exploratory and filtered queries were also spoken as requests
(*"could you find me some podcast episodes about …"*, *"are there any
podcasts that talk about …"*), which query understanding strips. On that
subset:

| phrasing | typed NDCG@10 | spoken | loss to transcription | WER |
|---|---|---|---|---|
| terse (*"gerrymandering explained"*) | 0.801 | 0.742 | 0.059 | 0.5% |
| conversational (*"play me an episode about gerrymandering explained"*) | 0.801 | 0.795 | **0.006** | 0.2% |

A sentence transcribes better than a bare phrase: the recognizer has
context, and it stops writing the query as a title, so the casing loss
disappears. Spoken as a request, these queries lose almost nothing to
transcription. The request framings themselves are stripped cleanly, 0.00
cost on 33 of 40.

Separately, query understanding has a cost of its own that the labeled
set exposes: typing the conversational text scores 0.801 against 0.881
for the labeled query with understanding skipped, and typing the terse
text with understanding on scores 0.906 against 0.936. In both cases the
cause is the same: the relevance judgments were made on the full query
text, and understanding removes words from it. For *"family friendly
history storytelling"* it sets the no-explicit filter and searches
*history storytelling* (0.91 NDCG@10 lost); *"kid friendly science
episodes"* becomes *science episodes* (0.86); *"science explainers under
30 minutes"* loses its duration phrase (0.72). The filter is applied, but
the words also carried lexical evidence the judges rewarded. This is a
property of typed search that the voice path inherits.

## Language

Six queries are in Spanish, French, German or Portuguese, spoken by
native library voices without a language hint. Detection got all six,
and transcription was exact except for the diacritics the labeled
queries lack (*Notícias de futebol em português* against *noticias de
futebol em portugues*), which cost 0.0–0.29 NDCG@10 each: lowercasing
does not recover it, accent folding would.

The reverse failure is worse. **Twenty-nine English clips were detected
as another language**, all of them a foreign-origin name spoken alone:
*omar suleiman* became *Omar Süleyman* (Turkish) in every voice and
*अमर सुलेमन* (Hindi) under noise; *dejan kovacevic* was heard as Polish,
Slovene, Bosnian or German. Those clips average 81% word error rate and
lose 0.55 NDCG@10. With no surrounding words, a short name is all the
language evidence the recognizer has. A language hint, which the API
accepts and the web app sends when the UI language is set, removes the
failure; detecting on a longer window or defaulting to the UI language
would remove it for everyone else.

## Confidence

The recognizer's mean token log-probability separates its failures: on
v1, −0.18 on clips it got exactly right, −0.34 on clips with errors, and
below −0.4 on most wrong names. It is stored with every voice search, so
a low-confidence transcript can be flagged in the UI, or routed to an
N-best or correction step, without any new signal.

## Latency

Base and conversational tiers (220 clips) through the production
configuration (lexical leg limited to 300 ms), one client, in sequence.
Speech server on the Apple GPU, Whisper small, greedy decoding.

| audio length | clips | transcription p50 / p95 ms | search after transcription p50 / p95 | end to end p50 / p95 |
|---|---|---|---|---|
| under 2 s | 59 | 361 / 457 | 326 / 574 | 687 / 1031 |
| 2–4 s | 130 | 373 / 486 | 298 / 445 | 671 / 931 |
| 4–8 s | 31 | 404 / 475 | 197 / 320 | 601 / 795 |
| **all** | 220 | **375 / 457** | **318 / 444** | **683 / 931** |

Transcription is about 0.4 s for any query under eight seconds (one
Whisper window, so the cost is nearly flat), and the search after it is
the text path measured in the serving report (318 ms p50, 444 ms p95,
with the lexical leg timing out on 94 of 220 at its 300 ms limit). The
text budget of 500 ms p95 excluding transcription holds on this run;
voice adds about 0.4 s on the GPU. On the CPU, under load, the same
transcription took 2–4 s; the GPU is not optional for interactive voice.
A distilled or int8 Whisper, or streaming while the user is still
speaking, are the ways down.

## Candidate mitigations, in order of expected value

1. **Normalize the transcript before search: lowercase, fold accents,
   treat hyphens as spaces.** Measured for lowercasing alone: recovers
   5.7 of the 8.7 points on the base tier and nearly all of the loss on
   error-free clips. Apply it to typed queries as well; it is the same
   bug.
2. **Send the language hint.** Fixes the 29 mis-detected English names
   outright; the API already accepts it.
3. **Catalog-aware entity correction.** Homophone near misses are the
   largest remaining loss. The transcript's n-grams can be matched
   against show, host and publisher names by phonetic key (Double
   Metaphone) or character edit distance under a threshold: *Lewis House
   → Lewis Howes* is a one-character edit, *Megan Kelly → Megyn Kelly* a
   phonetic match. Gate it on the recognizer's confidence.
4. **Keep filter phrases in the lexical text.** Fixes the understanding
   cost on filtered queries for voice and typed search alike.
5. **N-best transcripts** fused by RRF, so a name's second-best spelling
   gets a vote. Not measured; Whisper's beam search exposes it.
6. **Train with realistically corrupted queries**: the error distribution
   here (homophone substitutions on names, casing, dropped diacritics) is
   simple to simulate for the two-tower model's training queries.

## v1 versus v2: what the synthesizer changed

v1 was built with Chatterbox, a local model cloning macOS system voices.
Building it showed that it hallucinates on bare two-word texts (*joe
rogan* came out as *Plotkin*, *the daily wire* as a sentence of noise, in
about four of ten seeds), so every terse v1 clip was spoken inside a
carrier phrase, *"Search for joe rogan."*, which query understanding
strips. v2 uses ElevenLabs Eleven v4, speaks every query bare, and takes
its accents from voices of real speakers. Same queries, same tiers, same
degradations, same recognizer.

| | v1 Chatterbox | v2 ElevenLabs v4 |
|---|---|---|
| base WER / exact clips | 4.2% / 84% | 5.3% / 90% |
| base NEER | 23% | 22% |
| base loss, NDCG@10 | 0.082 | 0.087 |
| loss on error-free clips | 0.038 | 0.066 |
| accent spread of loss (best to worst voice) | 0.030–0.176 | 0.084–0.104 |
| en_IN loss | 0.176 | 0.094 |
| conversational loss to transcription | 0.044 | 0.006 |
| 0 dB noise loss | 0.389 | 0.360 |
| English clips detected as another language | 1 | 29 |

The headline is stable across synthesizers: about 8–9 points of NDCG@10,
dominated by names and casing, with the same names failing in both. Three
things changed, and each is a correction of v1, not a disagreement:

- **The accent finding was an artifact.** v1's "Indian English triples
  the error rate" was Chatterbox's rendering of one Apple voice, and it
  also broke the carrier phrase (*Touch for the ringer*). With real
  speakers' voices the accents are within two points of each other.
- **Bare names expose two failures the carrier hid.** Without
  surrounding words, the recognizer capitalizes the query as a title
  (error-free loss 0.066 against 0.038) and guesses the wrong language
  for foreign-origin names (29 clips against 1). Both are real: a user
  who says only a name into the microphone gets exactly this. The
  carrier phrase was masking them, which is why v1 looked better on
  error-free clips.
- **Conversational requests are nearly free in v2** because the
  recognizer, given a sentence, neither title-cases nor mis-detects.

v1 is kept for the record (`testset-v1.json`, `results-v1-*.json`); its
other reason for retirement is practical: Chatterbox pins torch 2.6 and
numpy 1.x, so it needs its own environment, and synthesizes at about five
seconds a clip against a few minutes for the whole v2 set.

## What this does not show

- **Synthetic speech.** No human recordings; the numbers are a lower
  bound. Twenty queries read by a handful of people would calibrate
  every table above, and would show whether the casing and
  language-detection behaviors hold for human voices (they should: they
  are recognizer behaviors, not voice behaviors).
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
- **ElevenLabs output is reproducible by (voice, model, seed)**, not bit
  for bit, and the hosted model can change; the manifest records all
  three for every clip, and the generated audio is kept locally.
