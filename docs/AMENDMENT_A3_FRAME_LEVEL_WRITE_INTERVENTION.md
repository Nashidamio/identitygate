# AMENDMENT A3 - FRAME-LEVEL PHYSICAL WRITE INTERVENTION

Date: 2026-09-10

Status: LOCK CANDIDATE - MUST BE COMMITTED BEFORE CLOSED-LOOP GATE RESULTS

Identifier note:
The supervisor dataset protocol already references Amendment A2 for the
enlarged-pool / Difficulty-Index amendment. This implementation amendment is
therefore A3 to avoid reusing that historical identifier.

## 1. Reason for amendment

The original IdentityGate implementation intent was per-object, per-frame
memory-write admission.

On the pinned frozen SAM3 VOS/PVS substrate, the following development-only
mechanism experiments now constrain the implementation:

- EXP025 verified that post-yield eviction of a non-conditioning frame from
  the global and per-object output dictionaries physically removes that frame
  before later tracking and changes downstream predictions.
- EXP026 rejected independent singleton tracker states because singleton
  execution was not exactly equivalent to batched vanilla B0.
- EXP027 rejected the memory_key_padding_mask attention-filter route because
  the active TransformerDecoderLayerv2.forward_pre path requires
  memory_key_padding_mask to be None.
- EXP028 rejected B=1 rowwise memory-fusion recomputation because the
  full-memory zero-omission control failed exact equivalence with batched B0.

Therefore no tested per-object intervention route preserves vanilla batched
SAM3 semantics exactly.

## 2. Supersession boundary

This amendment SUPERSEDES the requirement that the final physical memory
intervention itself must operate independently per object.

It DOES NOT supersede:

- the locked research question;
- frozen SAM3;
- object-level quality, temporal, self-identity, or relational-identity
  feature extraction;
- dual drift/theft labels;
- the B0, B1, B2, B3-S, B3-R, B5 ladder;
- video-clustered inference;
- matched-rate comparisons;
- full write-rate sweep curves;
- POR@30;
- J&F non-degradation;
- the whole-scene negative control;
- the one-touch TEST rule.

## 3. Closed-loop intervention unit

The physical intervention unit is one non-conditioning video frame.

Conditioning/prompt frames are always retained and are never gate-blocked.

For a gated variant v at frame t, the gate must produce one deterministic
admission score s_v(o,t) for every tracked object o using that variants frozen
feature set and frozen missingness handling.

Objects may not be silently omitted from aggregation because an identity
pointer or another optional feature is unavailable.

Define the frame score as:

    frame_score_v(t) = min_o s_v(o,t)

Define the physical decision as:

    ADMIT if frame_score_v(t) >= tau_v
    BLOCK otherwise

Thus the frame is admitted only when every tracked object passes the variants
object-level admission rule.

This is the fixed ALL-SAFE / minimum-score aggregation rule.

The aggregation rule may not be changed after this amendment is frozen.

## 4. Physical BLOCK semantics

A BLOCK decision uses the already-verified EXP025 mechanism:

1. allow the current frame prediction to complete;
2. after the predictor yields the current frame and before the next frame
   begins, remove the just-produced non-conditioning frame from the global
   memory-output dictionary;
3. remove the corresponding frame from the per-object non-conditioning output
   dictionaries;
4. retain predictor bookkeeping required for normal propagation.

This is physical whole-frame memory eviction.

Native SAM3 memory-selection behavior remains otherwise unchanged.

The method must not be described as per-object physical write blocking.

## 5. Causality

The frame-t gate decision may use only signals available from frame t and
earlier under the frozen gate definition.

No future GT, future predictions, future labels, or future outcome information
may affect the frame-t decision.

## 6. Frame-level write-rate denominator

For the amended closed-loop experiment:

    frame_write_rate =
        admitted eligible non-conditioning frame writes
        / all eligible non-conditioning frame-write opportunities

One processed eligible non-conditioning frame contributes exactly one
opportunity regardless of the number of tracked objects.

Conditioning/prompt frames are excluded from both numerator and denominator.

The denominator is therefore frame-level, not object-frame-level.

## 7. Matched-rate status

This amendment does NOT freeze the final matched-write-rate denominator,
reference condition, tolerance, or neutral budget-control policy.

The current physical fallback makes one non-conditioning frame the
intervention unit. Vanilla B0 physically retains every eligible frame under
this mechanism, so directly matching a gated method to B0 physical admission
rate would force a no-block operating point.

THESIS_RULES matched-write-rate open item therefore remains OPEN.

Before any headline matched-rate B1/B2/B3/B5 comparison, a separate protocol
must be frozen that resolves the reference/control budget without using
outcomes or TEST results. A random, periodic, or other outcome-independent
neutral budget selector may be considered, but the exact choice is not made
by this amendment.

Full POR-versus-write-rate, ITR-versus-write-rate, and unsafe-admission-versus-
write-rate curves remain mandatory.

Vanilla B0 remains the ungated practical reference.

No TEST data may be used to choose the matched-rate protocol.

## 8. Statistical path unchanged

The video remains the statistical cluster.

Primary inference remains video-clustered bootstrap.

POR@30 remains the primary recovery endpoint.

The minimum practically important hard-set POR effect remains +8 percentage
points.

J&F non-degradation and the whole-scene negative-control requirement remain in
force.

TEST is touched exactly once after all gate definitions, aggregation,
threshold-selection rules, comparisons, and preregistration artifacts are
frozen.

## 9. Claim boundary

The final method may be described as:

    frame-level physical memory-write admission derived from object-level
    quality, temporal, and identity signals.

It may not be described as:

    per-object physical memory-write blocking.

EXP026, EXP027, and EXP028 are retained as negative implementation evidence
rather than hidden or relaxed.
