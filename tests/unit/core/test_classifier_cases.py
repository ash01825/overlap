"""Table-driven test suite for OverlapClassifier (>= 60 cases, §7, §12).

Covers all required categories:
- backchannel only (11 Indic languages + English)
- haan + content
- silence-but-loud (acoustic prior)
- side-talk vs genuine quiet interruptions
- late partial (temporal progression)
- code-switch
- repeated 'haan haan haan' (within and exceeding bc_max_tokens)
- long 'theek hai ji' (within and exceeding bc_max_overlap_ms)
- questions, stops, contrasts, single vs multi-content tokens

Reports an accuracy/FIR/missed-interrupt table in test output.
Target: accuracy >= 95 %, FIR <= 5 %, missed-interrupt <= 5 %.
"""

from __future__ import annotations

from dataclasses import dataclass

from overlap.core.classifier import classify
from overlap.core.types import Label, OverlapFeatures, Tuning


@dataclass(frozen=True, slots=True)
class ClassifierTestCase:
    """A single labelled test case for classifier evaluation."""

    id: str
    category: str
    features: OverlapFeatures
    expected: Label
    description: str


CASES: tuple[ClassifierTestCase, ...] = (
    # =========================================================================
    # 1. Backchannel Only (11 languages + English)
    # =========================================================================
    ClassifierTestCase(
        id="bc_hi_haan",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="haan", lang="hi"
        ),
        expected=Label.BACKCHANNEL,
        description="Hindi 'haan' after bc_min_ms",
    ),
    ClassifierTestCase(
        id="bc_hi_haan_ji",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=250,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan ji",
            lang="hi",
        ),
        expected=Label.BACKCHANNEL,
        description="Hindi 'haan ji'",
    ),
    ClassifierTestCase(
        id="bc_hi_theek_hai",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=300,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="theek hai",
            lang="hi",
        ),
        expected=Label.BACKCHANNEL,
        description="Hindi 'theek hai'",
    ),
    ClassifierTestCase(
        id="bc_hi_achha",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=210, user_speaking=True, rel_loudness_db=0.0, partial_text="achha", lang="hi"
        ),
        expected=Label.BACKCHANNEL,
        description="Hindi 'achha'",
    ),
    ClassifierTestCase(
        id="bc_hi_hmm",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="hmm", lang="hi"
        ),
        expected=Label.BACKCHANNEL,
        description="Universal 'hmm'",
    ),
    ClassifierTestCase(
        id="bc_en_yeah",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="yeah", lang="en"
        ),
        expected=Label.BACKCHANNEL,
        description="English 'yeah'",
    ),
    ClassifierTestCase(
        id="bc_en_okay",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=230, user_speaking=True, rel_loudness_db=0.0, partial_text="okay", lang="en"
        ),
        expected=Label.BACKCHANNEL,
        description="English 'okay'",
    ),
    ClassifierTestCase(
        id="bc_en_right",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="right", lang="en"
        ),
        expected=Label.BACKCHANNEL,
        description="English 'right'",
    ),
    ClassifierTestCase(
        id="bc_en_mhm",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=210, user_speaking=True, rel_loudness_db=0.0, partial_text="mhm", lang="en"
        ),
        expected=Label.BACKCHANNEL,
        description="English 'mhm'",
    ),
    ClassifierTestCase(
        id="bc_en_uh_huh",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=240,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="uh-huh",
            lang="en",
        ),
        expected=Label.BACKCHANNEL,
        description="English 'uh-huh'",
    ),
    ClassifierTestCase(
        id="bc_ta_seri",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="seri", lang="ta"
        ),
        expected=Label.BACKCHANNEL,
        description="Tamil 'seri'",
    ),
    ClassifierTestCase(
        id="bc_ta_aama",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="aama", lang="ta"
        ),
        expected=Label.BACKCHANNEL,
        description="Tamil 'aama'",
    ),
    ClassifierTestCase(
        id="bc_te_avunu",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=230, user_speaking=True, rel_loudness_db=0.0, partial_text="avunu", lang="te"
        ),
        expected=Label.BACKCHANNEL,
        description="Telugu 'avunu'",
    ),
    ClassifierTestCase(
        id="bc_te_sare",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="sare", lang="te"
        ),
        expected=Label.BACKCHANNEL,
        description="Telugu 'sare'",
    ),
    ClassifierTestCase(
        id="bc_kn_howdu",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="howdu", lang="kn"
        ),
        expected=Label.BACKCHANNEL,
        description="Kannada 'howdu'",
    ),
    ClassifierTestCase(
        id="bc_ml_athe",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="athe", lang="ml"
        ),
        expected=Label.BACKCHANNEL,
        description="Malayalam 'athe'",
    ),
    ClassifierTestCase(
        id="bc_ml_sheri",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=230, user_speaking=True, rel_loudness_db=0.0, partial_text="sheri", lang="ml"
        ),
        expected=Label.BACKCHANNEL,
        description="Malayalam 'sheri'",
    ),
    ClassifierTestCase(
        id="bc_bn_hyan",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="hyan", lang="bn"
        ),
        expected=Label.BACKCHANNEL,
        description="Bengali 'hyan'",
    ),
    ClassifierTestCase(
        id="bc_bn_thik_ache",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=250,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="thik ache",
            lang="bn",
        ),
        expected=Label.BACKCHANNEL,
        description="Bengali 'thik ache'",
    ),
    ClassifierTestCase(
        id="bc_mr_ho",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=210, user_speaking=True, rel_loudness_db=0.0, partial_text="ho", lang="mr"
        ),
        expected=Label.BACKCHANNEL,
        description="Marathi 'ho'",
    ),
    ClassifierTestCase(
        id="bc_mr_barobar",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=240,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="barobar",
            lang="mr",
        ),
        expected=Label.BACKCHANNEL,
        description="Marathi 'barobar'",
    ),
    ClassifierTestCase(
        id="bc_gu_ha",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=210, user_speaking=True, rel_loudness_db=0.0, partial_text="ha", lang="gu"
        ),
        expected=Label.BACKCHANNEL,
        description="Gujarati 'ha'",
    ),
    ClassifierTestCase(
        id="bc_gu_saru",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220, user_speaking=True, rel_loudness_db=0.0, partial_text="saru", lang="gu"
        ),
        expected=Label.BACKCHANNEL,
        description="Gujarati 'saru'",
    ),
    ClassifierTestCase(
        id="bc_pa_haanji",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=220,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haanji",
            lang="pa",
        ),
        expected=Label.BACKCHANNEL,
        description="Punjabi 'haanji'",
    ),
    ClassifierTestCase(
        id="bc_pa_theek_aa",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=240,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="theek aa",
            lang="pa",
        ),
        expected=Label.BACKCHANNEL,
        description="Punjabi 'theek aa'",
    ),
    ClassifierTestCase(
        id="bc_early_ended",
        category="backchannel_only",
        features=OverlapFeatures(
            overlap_ms=120, user_speaking=False, rel_loudness_db=0.0, partial_text="haan", lang="hi"
        ),
        expected=Label.BACKCHANNEL,
        description="Short 'haan' where speech already ended before bc_min_ms",
    ),
    # =========================================================================
    # 2. Repeated 'haan haan haan' (limits & escalation)
    # =========================================================================
    ClassifierTestCase(
        id="rep_haan_3",
        category="repeated_backchannel",
        features=OverlapFeatures(
            overlap_ms=300,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan haan haan",
            lang="hi",
        ),
        expected=Label.BACKCHANNEL,
        description="3 repeated haans <= bc_max_tokens",
    ),
    ClassifierTestCase(
        id="rep_theek_2",
        category="repeated_backchannel",
        features=OverlapFeatures(
            overlap_ms=250,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="theek theek",
            lang="hi",
        ),
        expected=Label.BACKCHANNEL,
        description="2 repeated backchannels",
    ),
    ClassifierTestCase(
        id="rep_haan_4_escalated",
        category="repeated_backchannel",
        features=OverlapFeatures(
            overlap_ms=350,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan haan haan haan",
            lang="hi",
        ),
        expected=Label.INTERRUPT,
        description="4 repeated haans > bc_max_tokens (escalated)",
    ),
    ClassifierTestCase(
        id="rep_yeah_4_escalated",
        category="repeated_backchannel",
        features=OverlapFeatures(
            overlap_ms=380,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="yeah yeah yeah yeah",
            lang="en",
        ),
        expected=Label.INTERRUPT,
        description="4 repeated yeahs > bc_max_tokens (escalated)",
    ),
    # =========================================================================
    # 3. Long 'theek hai ji' (overlap duration escalation)
    # =========================================================================
    ClassifierTestCase(
        id="long_theek_hai_ji_normal",
        category="long_backchannel",
        features=OverlapFeatures(
            overlap_ms=350,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="theek hai ji",
            lang="hi",
        ),
        expected=Label.BACKCHANNEL,
        description="'theek hai ji' within normal overlap duration <= 600 ms",
    ),
    ClassifierTestCase(
        id="long_theek_hai_ji_escalated",
        category="long_backchannel",
        features=OverlapFeatures(
            overlap_ms=650,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="theek hai ji",
            lang="hi",
        ),
        expected=Label.INTERRUPT,
        description="Long 'theek hai ji' exceeding bc_max_overlap_ms (600 ms) escalates",
    ),
    ClassifierTestCase(
        id="long_haan_ji_escalated",
        category="long_backchannel",
        features=OverlapFeatures(
            overlap_ms=620,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan ji",
            lang="hi",
        ),
        expected=Label.INTERRUPT,
        description="Long 'haan ji' exceeding bc_max_overlap_ms escalates",
    ),
    # =========================================================================
    # 4. Haan + Content (Interruption & Escalation)
    # =========================================================================
    ClassifierTestCase(
        id="haan_content_suniye",
        category="haan_content",
        features=OverlapFeatures(
            overlap_ms=260,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan suniye",
            lang="hi",
        ),
        expected=Label.INTERRUPT,
        description="'haan suniye' with overlap >= content_min_overlap_ms",
    ),
    ClassifierTestCase(
        id="haan_content_doctor",
        category="haan_content",
        features=OverlapFeatures(
            overlap_ms=260,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan doctor",
            lang="hi",
        ),
        expected=Label.INTERRUPT,
        description="'haan doctor' with overlap >= content_min_overlap_ms",
    ),
    ClassifierTestCase(
        id="haan_stop_ruko",
        category="haan_content",
        features=OverlapFeatures(
            overlap_ms=80,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan ruko",
            lang="hi",
        ),
        expected=Label.INTERRUPT,
        description="'haan ruko' immediate stop word early exit",
    ),
    ClassifierTestCase(
        id="haan_stop_wait",
        category="haan_content",
        features=OverlapFeatures(
            overlap_ms=90,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan wait",
            lang="en",
        ),
        expected=Label.INTERRUPT,
        description="'haan wait' immediate stop word early exit",
    ),
    ClassifierTestCase(
        id="haan_contrast_lekin",
        category="haan_content",
        features=OverlapFeatures(
            overlap_ms=100,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan lekin",
            lang="hi",
        ),
        expected=Label.INTERRUPT,
        description="'haan lekin' contrast word early exit",
    ),
    ClassifierTestCase(
        id="haan_contrast_but",
        category="haan_content",
        features=OverlapFeatures(
            overlap_ms=90,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan but",
            lang="en",
        ),
        expected=Label.INTERRUPT,
        description="'haan but' contrast word early exit",
    ),
    ClassifierTestCase(
        id="haan_multi_content",
        category="haan_content",
        features=OverlapFeatures(
            overlap_ms=120,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan ek second",
            lang="hi",
        ),
        expected=Label.INTERRUPT,
        description="'haan ek second' with >= 2 content tokens decides early",
    ),
    ClassifierTestCase(
        id="haan_sentence",
        category="haan_content",
        features=OverlapFeatures(
            overlap_ms=150,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="haan ticket cancel karo",
            lang="hi",
        ),
        expected=Label.INTERRUPT,
        description="'haan ticket cancel karo' multiple content tokens",
    ),
    # =========================================================================
    # 5. Silence-but-Loud (Acoustic prior)
    # =========================================================================
    ClassifierTestCase(
        id="loud_assertive_speaking",
        category="silence_but_loud",
        features=OverlapFeatures(
            overlap_ms=260, user_speaking=True, rel_loudness_db=4.5, partial_text=None
        ),
        expected=Label.INTERRUPT,
        description="Acoustic prior: user speaking loudly (+4.5 dB >= assertive_db)",
    ),
    ClassifierTestCase(
        id="loud_assertive_speaking_empty_str",
        category="silence_but_loud",
        features=OverlapFeatures(
            overlap_ms=270, user_speaking=True, rel_loudness_db=5.0, partial_text=""
        ),
        expected=Label.INTERRUPT,
        description="Acoustic prior: user speaking loudly with empty partial string",
    ),
    ClassifierTestCase(
        id="loud_assertive_sustained",
        category="silence_but_loud",
        features=OverlapFeatures(
            overlap_ms=350, user_speaking=True, rel_loudness_db=6.0, partial_text=None
        ),
        expected=Label.INTERRUPT,
        description="Sustained loud user speech with no partial text",
    ),
    ClassifierTestCase(
        id="quiet_speech_ended_acoustic_bc",
        category="silence_but_loud",
        features=OverlapFeatures(
            overlap_ms=260, user_speaking=False, rel_loudness_db=0.0, partial_text=None
        ),
        expected=Label.BACKCHANNEL,
        description="Acoustic prior: short overlap + speech ended -> BACKCHANNEL",
    ),
    ClassifierTestCase(
        id="quiet_speech_ended_acoustic_bc_2",
        category="silence_but_loud",
        features=OverlapFeatures(
            overlap_ms=280, user_speaking=False, rel_loudness_db=-2.0, partial_text=""
        ),
        expected=Label.BACKCHANNEL,
        description="Acoustic prior: short overlap + speech ended with empty text",
    ),
    # =========================================================================
    # 6. Side-talk vs Genuine Quiet Interruption
    # =========================================================================
    ClassifierTestCase(
        id="sidetalk_water",
        category="side_talk",
        features=OverlapFeatures(
            overlap_ms=280,
            user_speaking=True,
            rel_loudness_db=-12.0,
            stt_conf=0.25,
            partial_text="ek glass paani",
        ),
        expected=Label.SIDE_TALK,
        description="Side-talk: quiet (-12 dB) with low confidence (0.25) and no lexicon hit",
    ),
    ClassifierTestCase(
        id="sidetalk_tea",
        category="side_talk",
        features=OverlapFeatures(
            overlap_ms=270,
            user_speaking=True,
            rel_loudness_db=-11.0,
            stt_conf=0.3,
            partial_text="chai le aao",
        ),
        expected=Label.SIDE_TALK,
        description="Side-talk: quiet (-11 dB) with low confidence (0.3)",
    ),
    ClassifierTestCase(
        id="sidetalk_door",
        category="side_talk",
        features=OverlapFeatures(
            overlap_ms=290,
            user_speaking=True,
            rel_loudness_db=-14.0,
            stt_conf=0.2,
            partial_text="darwaza band kar do",
        ),
        expected=Label.SIDE_TALK,
        description="Side-talk: far-field speech (-14 dB)",
    ),
    ClassifierTestCase(
        id="sidetalk_pen",
        category="side_talk",
        features=OverlapFeatures(
            overlap_ms=260,
            user_speaking=True,
            rel_loudness_db=-13.0,
            stt_conf=0.28,
            partial_text="can you pass the pen",
        ),
        expected=Label.SIDE_TALK,
        description="Side-talk in English: quiet with low confidence",
    ),
    ClassifierTestCase(
        id="quiet_interrupt_stop_word",
        category="side_talk",
        features=OverlapFeatures(
            overlap_ms=100,
            user_speaking=True,
            rel_loudness_db=-12.0,
            stt_conf=0.3,
            partial_text="ruko",
        ),
        expected=Label.INTERRUPT,
        description="Genuine quiet interruption with stop word must NOT be swallowed as side-talk",
    ),
    ClassifierTestCase(
        id="quiet_interrupt_stop_phrase",
        category="side_talk",
        features=OverlapFeatures(
            overlap_ms=110,
            user_speaking=True,
            rel_loudness_db=-10.0,
            stt_conf=0.4,
            partial_text="ek minute",
        ),
        expected=Label.INTERRUPT,
        description="Quiet interruption 'ek minute' must NOT be swallowed",
    ),
    ClassifierTestCase(
        id="quiet_interrupt_high_conf",
        category="side_talk",
        features=OverlapFeatures(
            overlap_ms=280,
            user_speaking=True,
            rel_loudness_db=-11.0,
            stt_conf=0.92,
            partial_text="please cancel my flight",
        ),
        expected=Label.INTERRUPT,
        description="Quiet interruption with high STT confidence is INTERRUPT, not SIDE_TALK",
    ),
    ClassifierTestCase(
        id="quiet_interrupt_high_conf_hi",
        category="side_talk",
        features=OverlapFeatures(
            overlap_ms=280,
            user_speaking=True,
            rel_loudness_db=-10.5,
            stt_conf=0.88,
            partial_text="suno mujhe bolne do",
        ),
        expected=Label.INTERRUPT,
        description="Quiet Hindi interruption with high confidence is INTERRUPT",
    ),
    # =========================================================================
    # 7. Late Partial & Temporal Progression
    # =========================================================================
    ClassifierTestCase(
        id="late_partial_early_wait",
        category="late_partial",
        features=OverlapFeatures(
            overlap_ms=100, user_speaking=True, rel_loudness_db=0.0, partial_text=None
        ),
        expected=Label.PENDING,
        description="No partial yet at 100 ms (< classify_min_ms) -> PENDING",
    ),
    ClassifierTestCase(
        id="late_partial_mid_wait",
        category="late_partial",
        features=OverlapFeatures(
            overlap_ms=180, user_speaking=True, rel_loudness_db=0.0, partial_text=None
        ),
        expected=Label.PENDING,
        description="No partial yet at 180 ms (< classify_min_ms) -> PENDING",
    ),
    ClassifierTestCase(
        id="late_partial_normal_loudness_wait",
        category="late_partial",
        features=OverlapFeatures(
            overlap_ms=260, user_speaking=True, rel_loudness_db=0.0, partial_text=None
        ),
        expected=Label.PENDING,
        description="No partial yet at 260 ms with normal loudness -> PENDING",
    ),
    ClassifierTestCase(
        id="late_partial_arrived_interrupt",
        category="late_partial",
        features=OverlapFeatures(
            overlap_ms=280, user_speaking=True, rel_loudness_db=0.0, partial_text="ek second"
        ),
        expected=Label.INTERRUPT,
        description="Partial arrives late with content tokens -> INTERRUPT",
    ),
    ClassifierTestCase(
        id="late_partial_arrived_bc",
        category="late_partial",
        features=OverlapFeatures(
            overlap_ms=280, user_speaking=True, rel_loudness_db=0.0, partial_text="theek hai"
        ),
        expected=Label.BACKCHANNEL,
        description="Partial arrives late with backchannel -> BACKCHANNEL",
    ),
    ClassifierTestCase(
        id="late_partial_cap_speaking_interrupt",
        category="late_partial",
        features=OverlapFeatures(
            overlap_ms=400, user_speaking=True, rel_loudness_db=0.0, partial_text=None
        ),
        expected=Label.INTERRUPT,
        description="Hard cap reached at 400 ms with user still speaking -> failsafe INTERRUPT",
    ),
    ClassifierTestCase(
        id="late_partial_cap_ended_bc",
        category="late_partial",
        features=OverlapFeatures(
            overlap_ms=400, user_speaking=False, rel_loudness_db=0.0, partial_text=None
        ),
        expected=Label.BACKCHANNEL,
        description="Hard cap reached at 400 ms with speech ended -> failsafe BACKCHANNEL",
    ),
    # =========================================================================
    # 8. Code-Switching
    # =========================================================================
    ClassifierTestCase(
        id="code_switch_yeah_theek_hai",
        category="code_switch",
        features=OverlapFeatures(
            overlap_ms=250, user_speaking=True, rel_loudness_db=0.0, partial_text="yeah theek hai"
        ),
        expected=Label.BACKCHANNEL,
        description="English 'yeah' + Hindi 'theek hai' is pure backchannel",
    ),
    ClassifierTestCase(
        id="code_switch_right_accha",
        category="code_switch",
        features=OverlapFeatures(
            overlap_ms=240, user_speaking=True, rel_loudness_db=0.0, partial_text="right accha"
        ),
        expected=Label.BACKCHANNEL,
        description="English 'right' + Hindi 'accha' is pure backchannel",
    ),
    ClassifierTestCase(
        id="code_switch_wait_ek_minute",
        category="code_switch",
        features=OverlapFeatures(
            overlap_ms=80, user_speaking=True, rel_loudness_db=0.0, partial_text="wait ek minute"
        ),
        expected=Label.INTERRUPT,
        description="English 'wait' + Hindi 'ek minute' stop words",
    ),
    ClassifierTestCase(
        id="code_switch_okay_but_listen",
        category="code_switch",
        features=OverlapFeatures(
            overlap_ms=100, user_speaking=True, rel_loudness_db=0.0, partial_text="okay but listen"
        ),
        expected=Label.INTERRUPT,
        description="English 'okay but listen' with contrast word 'but'",
    ),
    ClassifierTestCase(
        id="code_switch_okay_samajh_gaya",
        category="code_switch",
        features=OverlapFeatures(
            overlap_ms=260, user_speaking=True, rel_loudness_db=0.0, partial_text="okay samajh gaya"
        ),
        expected=Label.INTERRUPT,
        description="Code-switch with >= 2 content tokens decides INTERRUPT",
    ),
    ClassifierTestCase(
        id="code_switch_haan_cancel",
        category="code_switch",
        features=OverlapFeatures(
            overlap_ms=260, user_speaking=True, rel_loudness_db=0.0, partial_text="haan cancel"
        ),
        expected=Label.INTERRUPT,
        description="Hindi 'haan' + English content 'cancel' at content_min_overlap_ms",
    ),
    # =========================================================================
    # 9. Question Words and Punctuation
    # =========================================================================
    ClassifierTestCase(
        id="q_hi_kya",
        category="question",
        features=OverlapFeatures(
            overlap_ms=70, user_speaking=True, rel_loudness_db=0.0, partial_text="kya", lang="hi"
        ),
        expected=Label.INTERRUPT,
        description="Hindi 'kya' question word immediate interrupt",
    ),
    ClassifierTestCase(
        id="q_en_why_mark",
        category="question",
        features=OverlapFeatures(
            overlap_ms=60, user_speaking=True, rel_loudness_db=0.0, partial_text="why?", lang="en"
        ),
        expected=Label.INTERRUPT,
        description="Question mark immediate interrupt",
    ),
    ClassifierTestCase(
        id="q_hi_kab",
        category="question",
        features=OverlapFeatures(
            overlap_ms=90,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="kab start hoga",
            lang="hi",
        ),
        expected=Label.INTERRUPT,
        description="Hindi 'kab' question",
    ),
    ClassifierTestCase(
        id="q_ta_eppadi",
        category="question",
        features=OverlapFeatures(
            overlap_ms=80,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="eppadi irukku",
            lang="ta",
        ),
        expected=Label.INTERRUPT,
        description="Tamil 'eppadi' question",
    ),
    ClassifierTestCase(
        id="q_te_entha",
        category="question",
        features=OverlapFeatures(
            overlap_ms=80,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="entha train",
            lang="te",
        ),
        expected=Label.INTERRUPT,
        description="Telugu 'entha' question",
    ),
    ClassifierTestCase(
        id="q_en_how_much",
        category="question",
        features=OverlapFeatures(
            overlap_ms=90,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="how much is it",
            lang="en",
        ),
        expected=Label.INTERRUPT,
        description="English 'how much' question",
    ),
    # =========================================================================
    # 10. Stop and Contrast Words
    # =========================================================================
    ClassifierTestCase(
        id="stop_ruko",
        category="stop_contrast",
        features=OverlapFeatures(
            overlap_ms=50, user_speaking=True, rel_loudness_db=0.0, partial_text="ruko"
        ),
        expected=Label.INTERRUPT,
        description="Stop word 'ruko'",
    ),
    ClassifierTestCase(
        id="stop_nahi",
        category="stop_contrast",
        features=OverlapFeatures(
            overlap_ms=50, user_speaking=True, rel_loudness_db=0.0, partial_text="nahi"
        ),
        expected=Label.INTERRUPT,
        description="Stop word 'nahi'",
    ),
    ClassifierTestCase(
        id="stop_bas",
        category="stop_contrast",
        features=OverlapFeatures(
            overlap_ms=50, user_speaking=True, rel_loudness_db=0.0, partial_text="bas"
        ),
        expected=Label.INTERRUPT,
        description="Stop word 'bas'",
    ),
    ClassifierTestCase(
        id="stop_en_stop",
        category="stop_contrast",
        features=OverlapFeatures(
            overlap_ms=50, user_speaking=True, rel_loudness_db=0.0, partial_text="stop"
        ),
        expected=Label.INTERRUPT,
        description="Stop word 'stop'",
    ),
    ClassifierTestCase(
        id="stop_en_wait",
        category="stop_contrast",
        features=OverlapFeatures(
            overlap_ms=50, user_speaking=True, rel_loudness_db=0.0, partial_text="wait"
        ),
        expected=Label.INTERRUPT,
        description="Stop word 'wait'",
    ),
    ClassifierTestCase(
        id="contrast_lekin",
        category="stop_contrast",
        features=OverlapFeatures(
            overlap_ms=60, user_speaking=True, rel_loudness_db=0.0, partial_text="lekin"
        ),
        expected=Label.INTERRUPT,
        description="Contrast word 'lekin'",
    ),
    ClassifierTestCase(
        id="contrast_par",
        category="stop_contrast",
        features=OverlapFeatures(
            overlap_ms=60, user_speaking=True, rel_loudness_db=0.0, partial_text="par"
        ),
        expected=Label.INTERRUPT,
        description="Contrast word 'par'",
    ),
    ClassifierTestCase(
        id="contrast_magar",
        category="stop_contrast",
        features=OverlapFeatures(
            overlap_ms=60, user_speaking=True, rel_loudness_db=0.0, partial_text="magar"
        ),
        expected=Label.INTERRUPT,
        description="Contrast word 'magar'",
    ),
    ClassifierTestCase(
        id="contrast_en_but",
        category="stop_contrast",
        features=OverlapFeatures(
            overlap_ms=60, user_speaking=True, rel_loudness_db=0.0, partial_text="but"
        ),
        expected=Label.INTERRUPT,
        description="Contrast word 'but'",
    ),
    ClassifierTestCase(
        id="contrast_ta_aanaal",
        category="stop_contrast",
        features=OverlapFeatures(
            overlap_ms=60, user_speaking=True, rel_loudness_db=0.0, partial_text="aanaal", lang="ta"
        ),
        expected=Label.INTERRUPT,
        description="Tamil contrast word 'aanaal'",
    ),
    # =========================================================================
    # 11. Content Tokens (Early vs Late)
    # =========================================================================
    ClassifierTestCase(
        id="content_flight_ticket",
        category="content_tokens",
        features=OverlapFeatures(
            overlap_ms=100, user_speaking=True, rel_loudness_db=0.0, partial_text="flight ticket"
        ),
        expected=Label.INTERRUPT,
        description="2 content tokens early exit before 250 ms",
    ),
    ClassifierTestCase(
        id="content_cancel_ticket",
        category="content_tokens",
        features=OverlapFeatures(
            overlap_ms=90, user_speaking=True, rel_loudness_db=0.0, partial_text="cancel ticket"
        ),
        expected=Label.INTERRUPT,
        description="2 content tokens early exit before 250 ms",
    ),
    ClassifierTestCase(
        id="content_account_balance",
        category="content_tokens",
        features=OverlapFeatures(
            overlap_ms=120,
            user_speaking=True,
            rel_loudness_db=0.0,
            partial_text="account balance check",
        ),
        expected=Label.INTERRUPT,
        description="3 content tokens early exit",
    ),
    ClassifierTestCase(
        id="content_single_early",
        category="content_tokens",
        features=OverlapFeatures(
            overlap_ms=120, user_speaking=True, rel_loudness_db=0.0, partial_text="cancel"
        ),
        expected=Label.PENDING,
        description="1 content token before content_min_overlap_ms -> PENDING",
    ),
    ClassifierTestCase(
        id="content_single_late",
        category="content_tokens",
        features=OverlapFeatures(
            overlap_ms=260, user_speaking=True, rel_loudness_db=0.0, partial_text="cancel"
        ),
        expected=Label.INTERRUPT,
        description="1 content token at content_min_overlap_ms -> INTERRUPT",
    ),
)


def test_classifier_cases_table_accuracy() -> None:
    """Evaluate classifier over the built-in test suite and print accuracy/FIR/missed table.

    Asserts:
    - total cases >= 60
    - accuracy >= 95 %
    - False Interruption Rate (FIR) <= 5 %
    - Missed Interruption Rate <= 5 %
    """
    assert len(CASES) >= 60, f"Expected >= 60 cases, got {len(CASES)}"

    tuning = Tuning()
    total = len(CASES)
    correct = 0

    # False Interruption Rate: classified as INTERRUPT when expected was BACKCHANNEL or SIDE_TALK
    false_interrupts = 0
    non_interrupt_count = 0

    # Missed Interruption Rate: classified as NOT INTERRUPT when expected was INTERRUPT
    missed_interrupts = 0
    interrupt_count = 0

    failures: list[str] = []

    for case in CASES:
        verdict = classify(case.features, tuning)
        is_correct = verdict.label == case.expected
        if is_correct:
            correct += 1
        else:
            failures.append(
                f"[{case.id}] ({case.category}) '{case.description}': "
                f"expected {case.expected.value}, got {verdict.label.value} "
                f"(reason: {verdict.reason})"
            )

        if case.expected in (Label.BACKCHANNEL, Label.SIDE_TALK):
            non_interrupt_count += 1
            if verdict.label == Label.INTERRUPT:
                false_interrupts += 1

        if case.expected == Label.INTERRUPT:
            interrupt_count += 1
            if verdict.label != Label.INTERRUPT:
                missed_interrupts += 1

    accuracy_pct = (correct / total) * 100.0
    fir_pct = (false_interrupts / non_interrupt_count * 100.0) if non_interrupt_count > 0 else 0.0
    missed_pct = (missed_interrupts / interrupt_count * 100.0) if interrupt_count > 0 else 0.0

    # Format table output for telemetry and test logs
    report = (
        f"\n{'=' * 70}\n"
        f"OVERLAP CLASSIFIER EVALUATION REPORT\n"
        f"{'=' * 70}\n"
        f"Total Cases Evaluated:       {total}\n"
        f"Correct Decisions:           {correct}\n"
        f"Overall Accuracy:            {accuracy_pct:.2f} % (Target >= 95.0 %)\n"
        f"False Interruption Rate:     {fir_pct:.2f} % (Target <= 5.0 %)\n"
        f"Missed Interruption Rate:    {missed_pct:.2f} % (Target <= 5.0 %)\n"
        f"Non-Interrupt Ground Truths: {non_interrupt_count}\n"
        f"Interrupt Ground Truths:     {interrupt_count}\n"
        f"{'=' * 70}\n"
    )
    print(report)

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")

    assert accuracy_pct >= 95.0, f"Accuracy {accuracy_pct:.2f}% < 95.0%"
    assert fir_pct <= 5.0, f"FIR {fir_pct:.2f}% > 5.0%"
    assert missed_pct <= 5.0, f"Missed interruptions {missed_pct:.2f}% > 5.0%"
