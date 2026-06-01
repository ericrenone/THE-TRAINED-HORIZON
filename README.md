# THE TRAINED HORIZON

On Peer Review AI, Citation Homophily, and How the Recursive Training of Academic Evaluation Systems Converts the Paradigm's Blind Spot into Computational Preference

*Eric Ren · ERI Labs · Jersey City, New Jersey · May 2026 · github.com/ericrenone*

---

> "AI models collapse when trained on recursively self-generated data... Over successive generations, the tails of the original data distribution are irreversibly lost." — Shumailov et al., *Nature*, 2024

> "The advantage of noise reduction is only a genuine advantage when the mean is in the right place. Consistent application of a biased standard is worse than noisy application of a correct one." — Kahneman, Sibony & Sunstein, *Noise: A Flaw in Human Judgment*, 2021

> "For unto every one that hath shall be given, and he shall have abundance: but from him that hath not shall be taken away even that which he hath." — Matthew 25:29 [cited as 'The Matthew Effect in Science' in Merton, *Science*, 1968]

> "The planner over-invests in skills destined for obsolescence, a distortion that increases monotonically with AI prevalence." — Peterson, University of Poitiers, arXiv:2508.19625, 2025

---

## The Argument in One Paragraph

The prior documents in this lineage establish a precise diagnosis: the academic prestige apparatus is calibrated for paradigm depth (Ψ), not inter-paradigm translation capacity (B), and this miscalibration constitutes a compound multiplicative filter that systematically suppresses the translation-architect researchers uniquely positioned to unlock the field's hardest problems. That analysis was correct and incomplete. It omitted the development that converts the compound filter from a correctable institutional defect into a self-reinforcing computational process: the deployment of AI systems — trained on the paradigm-internal outputs of the already-failing apparatus — into the evaluation roles of peer review, grant scoring, and literature discovery. These systems do not merely inherit the apparatus's blind spot. They encode it as a trained preference, eliminate the human review variance that occasionally allowed translation-architect work to survive, and generate the training corpus for their own successors from the filtered outputs they produce. The result is the exact analog of model collapse applied to the research evaluation domain: each generation of the AI evaluation apparatus learns from a distribution from which the prior generation removed more of the translation-architect tail, until the tail is no longer present at sufficient rates to influence the model's learned preferences. The trained horizon — the paradigm's boundary encoded as the evaluator's preference boundary — is not static. It converges.

---

## Table of Contents

1. The Evaluator Arrives
2. What Was in the Training Data
3. The Variance That Used to Protect
4. Citation Homophily as Quality Signal
5. The Recursive Degradation: Model Collapse in the Evaluation Apparatus
6. The Three Deployment Vectors
7. Formal Extension of the Compound Filter
8. What This Means for the Three Locked Rooms
9. The Identification Problem: You Cannot Calibrate from Filtered Data
10. Novel Theoretical Coverage
11. Falsifiable Predictions
12. Complete Evidence Base

---

## 1. The Evaluator Arrives

In the autumn of 2024, a paper appeared in *Nature* with an unusual structure. Its authors — Ilia Shumailov and colleagues — were studying what happens to AI language models trained iteratively on their own outputs rather than on original human-generated data. They called the phenomenon model collapse: a degenerative process in which successive training generations cause models to progressively forget the tails of the distribution they were originally trained on. The early-generation model, trained on human data, represents rare and anomalous examples at nonzero probability. The next-generation model, trained on the prior model's outputs, underrepresents those rare examples because the model generates them at lower probability. Each successive generation underrepresents the tail further, until it is effectively absent from the training distribution. The model has forgotten what used to be possible at the distribution's edge.

The paper was submitted to *Nature*, reviewed by human experts, revised, and published. Its cross-paradigm structure — drawing on information theory, statistical mechanics, and machine learning simultaneously to characterize a phenomenon that ML researchers had not previously formalized — required reviewers capable of evaluating all three. Human review panels have variance. The panel for this paper included, one infers, reviewers with the necessary cross-paradigm competence.

Here is the thought experiment this paper makes unavoidable.

If the review of that paper had been assisted by an AI reviewer-matching system trained on the citation networks of established ML research — as multiple major ML venues now deploy or are actively piloting — would the same reviewers have been matched? A system trained on citation adjacency in the ML paradigm would have found the paper's information-theoretic and statistical-physics vocabulary anomalous relative to its training distribution of ML papers. It would have matched ML reviewers, who could evaluate the machine learning claims but not the information-theoretic derivations. The cross-paradigm synthesis — the specific mechanism that made the paper's contribution genuine — would have been evaluated by reviewers constitutionally positioned to miss it.

The paper documenting model collapse in AI systems might itself have been an early casualty of the mechanism it documented.

This is not a historical claim about what happened to that specific paper. It is a structural observation about what is now systematically happening to the papers that most need to happen — and about the direction in which the situation is moving with each deployment generation.

---

## 2. What Was in the Training Data

Every AI system deployed in academic evaluation is trained on something. The relevant question — never fully posed by the institutions deploying these systems — is: trained on what?

The training corpus for any AI evaluation tool in the research domain is, at its core, a corpus of research that prior evaluation systems found acceptable. Grant proposals that funding agencies chose to fund. Papers that human review panels chose to accept. Reviewer-paper matches that historically produced positive outcomes, as measured by subsequent citation count — a paradigm-internal signal. Literature discovery systems trained on citation graphs, which are by construction the accumulated record of what paradigm-internal researchers cited when writing paradigm-internal papers.

Each of these training sources shares a structural property: it was generated by a process that had already applied the compound filter. The accepted papers are the papers that survived six stages of selection, each of which the prior documents in this lineage have shown to be calibrated for paradigm depth (Ψ) and blind to translation capacity (B). The citation graph encodes the reading habits of researchers trained within their paradigms, citing the literature their advisors directed them toward.

There is no translation-architect signal in this training data — not because translation-architect work does not exist, but because the filter already removed most of it before the data was generated. What remains is the paradigm's preferred output. The AI system learns that output as its preference. Its preference is not biased relative to what it was trained on. It is precisely accurate with respect to what it was trained on. The problem is not the AI system's failure to learn correctly. The problem is that what it correctly learned is the wrong distribution.

---

## 3. The Variance That Used to Protect

This is the document's core formal contribution.

The compound filter analyzed in the prior documents operated on translation-architect work through six institutional stages. Each stage had a selection probability less than one. But each stage was also operated by humans — and humans, even when their mean judgment is biased against translation-architect work, have variance in their judgments.

The variance was not merely noise. For translation-architect papers, variance was structural protection.

Consider a three-reviewer panel evaluating a translation-architect paper P_T. The panel's composition is determined by a human program chair applying paradigm-internal matching, whose matching is imperfect — an occasional error in reviewer assignment, an occasional inclusion of someone who has read across the relevant fields.

Let E_i(P_T) denote the evaluation score assigned by reviewer *i* to paper P_T. Suppose the distribution of E_i(P_T) across the human reviewer pool has mean μ_T below the acceptance threshold θ — because the pool is paradigm-depth-calibrated — but variance σ²_T > 0, because individual reviewers vary in their exposure to cross-paradigm work.

The probability that P_T is accepted is not simply P(μ_T > θ). It is a function of the aggregate scoring rule combined with the variance:

```
P(accept | human panel) = P(aggregate{E₁, E₂, E₃} > θ) > 0
```

For translation-architect papers where at least one reviewer has cross-paradigm expertise, this probability is non-trivial. The paper survives because one reviewer's strong positive signal is sufficient to overcome two moderate negative signals in a discussion process. The human variance is the mechanism.

Now introduce an AI evaluation system E_a trained on the historical corpus of accepted papers — which, as established in Section 2, is a distribution from which most translation-architect papers were already removed by the prior filter.

E_a(P_T) is not a draw from a distribution with variance σ²_T. It is a near-deterministic function of the paper's features as measured against the AI's learned preference distribution. The system produces consistent scores. Its consistency is its most marketed property. Review fatigue, in-group bias, reviewer prestige effects — the variances the AI eliminates are real problems in human review. But they are not the only source of variance. The variance the AI also eliminates — silently, without acknowledgment — is the reviewer who happened to have read the algebraic geometry that the translation-architect paper requires.

Formally:

```
P(accept | AI evaluation) = P(E_a(P_T) > θ) ≈ P(μ_T > θ) ≈ 0
```

because E_a(P_T) ≈ E[E_h(P_T)] = μ_T < θ and Var[E_a(P_T)] ≈ 0.

The rare event — the one-in-twenty review panel that included a cross-paradigm expert who could recognize the translation-architect paper — is not a random fluctuation to be eliminated. It is the mechanism by which the field's most consequential work occasionally survived the apparatus. AI evaluation eliminates it with perfect fidelity to its training objective.

The consistency is the damage.

---

## 4. Citation Homophily as Quality Signal

Citation networks are paradigm-structured by construction. Work within a paradigm cites prior work within the same paradigm — because the paradigm defines what counts as the relevant prior literature. The review process instructs authors to cite their fields. The field is the paradigm. The result is a citation network with dense within-paradigm clusters and sparse between-paradigm connections.

Translation-architect work is defined, in the ERIE framework, by the structural property that its citation profile spans two paradigms: it cites from both Φ_i (the native paradigm whose problem it is solving) and Φ_j (the foreign paradigm whose tools it imports). This cross-cluster citation pattern is not a deficiency in the paper's bibliographic practice. It is the bibliographic signature of inter-paradigm synthesis.

But it is also a bibliographic anomaly relative to every AI system trained on the paradigm-internal citation graph.

Consider how AI evaluation systems use citation signals. Literature discovery systems — Semantic Scholar's AI recommendations, Google Scholar's suggestions, connected-paper graphs — use embedding-based similarity trained on citation co-occurrence. Two papers are considered similar if they cite similar papers and are cited by similar papers. Translation-architect work, citing from two distinct citation clusters, is maximally distant from both clusters' centroids in the embedding space. The system locates it in a low-density region of the embedding — which a system trained on density as a quality proxy reads as low quality or low relevance.

Reviewer matching systems use keyword extraction and citation network proximity to identify reviewers whose expertise matches a submission. A translation-architect paper's keywords span two paradigms. The system has two matching strategies: match on Paradigm A keywords (producing reviewers who can evaluate half the paper) or match on Paradigm B keywords (producing reviewers who can evaluate the other half). The synthesis — the specific operation the paper is performing — requires a reviewer who can evaluate both simultaneously. The system has no mechanism for this match because the training data contains no positive examples of reviewers successfully evaluating cross-paradigm synthesis at the contributory expertise level.

Grant scoring systems trained on funded proposals will have learned that funded proposals use vocabulary coherent with a single paradigm's technical community. A cross-paradigm proposal will contain vocabulary from two technical communities and pattern-match to "insufficiently focused" — not because it is, but because the system has no positive training examples of funded cross-paradigm proposals against which to calibrate.

In each case, citation homophily in the training data is learned as a positive quality signal. Cross-paradigm citation patterns are learned as negative quality signals — not through any explicit instruction, but as the correct representation of what the training distribution contained.

---

## 5. The Recursive Degradation: Model Collapse in the Evaluation Apparatus

Shumailov et al. (2024) identify three properties of model collapse in language models trained on their own outputs. Early-generation models trained on human data produce outputs that include rare examples from the tails of the human distribution — at low probability, but nonzero. Models trained on those outputs produce the rare examples at lower probability still, because the training data underrepresents them. After sufficient generations, the rare examples are absent from the model's output distribution entirely — not because the model was instructed to exclude them, but because they fell below the threshold of effective representation in the training data.

The analog in the research evaluation apparatus is structurally exact.

**Generation 0.** The human review apparatus operates with variance. Occasionally, a translation-architect paper P_T survives. The body of published literature contains P_T at a low but nonzero rate *r₀ > 0*, representing the cases where human variance was sufficient to overcome the paradigm-depth calibration.

**Generation 1.** An AI evaluation tool E₁ is trained on the published literature from Generation 0, which contains P_T at rate *r₀*. The tool learns to score translation-architect papers at μ_T, the paradigm-depth-calibrated mean, which is below the acceptance threshold. A small fraction of P_T papers still pass — those at the high end of E₁'s distribution. The rate of P_T in the published literature drops: *r₁ < r₀*.

**Generation 2.** E₂ is trained on the published literature from Generation 1, which contains P_T at rate *r₁ < r₀*. E₂ has fewer positive training examples of translation-architect work than E₁. Its learned preference distribution is more paradigm-depth-concentrated. The rate drops further: *r₂ < r₁*.

**Generation n.** As *n* grows, *rₙ → 0*. The AI evaluation apparatus has forgotten that translation-architect work is possible, because the training corpus from which it learns the distribution of acceptable research no longer contains translation-architect work at rates sufficient to influence the model's learned preferences.

The Erdős breakthrough required that algebraic number theory existed in a form that the next breakthrough-maker could find. If the literature discovery apparatus has encoded the paradigm's citation graph as a preference model that systematically fails to surface algebraic number theory to combinatorial geometers, the conditions for the next Erdős moment are being computationally suppressed in the generation before anyone noticed the suppression was occurring.

This is not a concern about a distant future. The first AI reviewer-matching systems were deployed at major ML conferences beginning in 2022-2024. The generation cycle for large language models used in review assistance is annual or shorter. The recursive degradation is not a theoretical extrapolation. It is already in its early iterations.

---

## 6. The Three Deployment Vectors

The recursive degradation operates through three specific channels in current academic infrastructure. Each has a distinct mechanism and a distinct contribution to the compound filter's computational extension.

**Vector 1: Reviewer Matching**

OpenReview, the platform used by ICLR, NeurIPS, ICML, and multiple other major ML conferences, integrated AI-assisted reviewer-paper matching beginning in its early versions and has deepened this integration as LLM capabilities improved. The matching system operates on semantic similarity between reviewer expertise profiles — derived from publication history and stated areas — and submitted paper content.

Translation-architect papers have a structural mismatch with this architecture. Their content spans two expertise domains. The system must anchor on one or the other. In practice, it anchors on the domain in which the paper's submission category places it: an ML conference receives an ML paper, so the system matches ML reviewers. The foreign paradigm's vocabulary — the algebraic geometry, the formal verification, the statistical physics — is present in the paper but absent from the reviewer pool the conference maintains.

The result is not that the translation-architect paper receives hostile reviews. It receives reviews from experts who can evaluate the native-paradigm components competently and who genuinely lack the tools to evaluate the foreign-paradigm components. Their reviews will raise concerns about rigor or relevance in precisely the foreign-paradigm sections — concerns that are accurate from within their own expertise and structurally unable to assess the synthesis that the foreign-paradigm tools enable.

**Vector 2: Grant Scoring**

The U.S. National Institutes of Health, the National Science Foundation, and multiple international funding agencies have explored or piloted AI assistance in grant proposal review. The motivation is sound: reviewer fatigue, volume overload, and consistency across review panels are genuine problems. AI pre-scoring of proposals — with human reviewers then focusing on the top-scored applications — offers a real efficiency gain.

But the efficiency gain is achieved by eliminating the review stage that, for translation-architect proposals, most mattered: the human stage where variance occasionally included a reviewer who could recognize cross-paradigm synthesis. If AI pre-scoring consistently places translation-architect proposals below the threshold for human review, those proposals never reach the human stage in which variance-based protection might have operated.

The pre-scoring system converts the compound filter's probabilistic structure into a hard cutoff. Translation-architect proposals that would have had nonzero probability of survival under fully human review now face the near-zero probability implied by an AI score trained on paradigm-internal funded proposal distributions.

**Vector 3: Literature Discovery**

The upstream effect — less discussed but potentially most consequential — operates at the literature discovery stage, before papers are written, submitted, or reviewed.

Researchers use AI-powered literature discovery tools to identify relevant prior work. Semantic Scholar's 200-million-paper database uses AI to recommend related papers. Research planning tools built on large language models synthesize relevant literature in response to queries. These tools are trained on citation graphs and abstracts and surface literature based on embedding-space proximity.

For a researcher at the paradigm boundary — a machine learning researcher who has recognized that algebraic geometry might be relevant to the generalization problem — the literature discovery tool's recommendations will be anchored to ML papers discussing the generalization problem, not to algebraic geometry papers providing the tools. The embedding-space distance between the generalization problem literature and the algebraic geometry literature is large by construction: the two communities do not cite each other and do not use similar vocabulary.

The researcher who might have made the paradigm-crossing recognition is not served the algebraic geometry literature. They are served more ML literature on the generalization problem. The cross-paradigm abduction — the recognition that a different room's tools are relevant — is suppressed at the literature discovery stage, before the paper is written, before the review process begins, before any of the prior compound filter's stages are even reached.

---

## 7. Formal Extension of the Compound Filter

The compound filter from the prior documents in this lineage was characterized as six multiplicative stages, each with selection probability *pᵢ* < 1:

```
P(translation-architect survives) = ∏ᵢ₌₁⁶ pᵢ ≪ 1
```

The AI evaluation apparatus introduces a seventh stage with structural properties that make it qualitatively different from the prior six.

Let E_t be the AI evaluation system trained at generation *t*. Define:

```
p₇(t) = P(translation-architect paper P_T passes E_t's filter)
```

Stage 7 has three properties not shared by stages 1–6.

**Property A: Variance suppression.** Human stages had variance σ² > 0 in their evaluation of P_T. Stage 7 has σ² ≈ 0. The tail-event protection that variance afforded is eliminated.

**Property B: Recursive training.** The corpus C_{t+1} from which E_{t+1} is trained is the filtered output of E_t applied to C_t. As *t* increases:

```
{P_T in C_{t+1}} ⊆ {P_T in C_t}   [set inclusion, strictly smaller]
```

Therefore p₇(t+1) ≤ p₇(t), with equality only if p₇(t) = 0 (full collapse) or if the feedback loop is interrupted by external calibration.

**Property C: Scope expansion.** Stages 1–6 applied to specific decision points: doctoral admission, publication acceptance, hiring. Stage 7 is being integrated into all three simultaneously — literature discovery (pre-Stage 1), reviewer matching (Stage 4), grant scoring (Stage 3), and citation-based hiring signals (Stage 5). The AI evaluation apparatus is not adding one stage to the compound filter. It is adding an automated layer beneath every prior stage.

The compound filter probability becomes:

```
P(translation-architect survives, generation t) = [∏ᵢ₌₁⁶ pᵢ] × p₇(t)
```

As *t* increases, p₇(t) → 0, driving the total probability toward zero faster than any institutional reform of stages 1–6 could compensate.

Define Silo_apparatus(t), the apparatus-level silo measure at training generation *t*:

```
Silo_apparatus(t) = 1 − P(E_t correctly calibrates a translation-architect paper)
```

The prior documents establish that Silo_apparatus(t=0) — the human apparatus's silo measure — is already very high. The trained horizon prediction: Silo_apparatus(t) is strictly increasing in *t* under current deployment conditions, and the rate of increase exceeds what any concurrent institutional reform could offset if the recursive training structure is not interrupted.

The silo measure for the apparatus as a whole:

```
Silo(E_t) > Silo(human apparatus)   for all t > 0
```

not because the AI system is less capable than its human predecessor, but because it is more consistent — and consistency, applied to a biased mean, eliminates the variance that protected the tail.

---

## 8. What This Means for the Three Locked Rooms

The three locked rooms identified in the prior documents of this lineage — the generalization proof, formal technical alignment, and the consciousness-interpretability convergence — are precisely the research programs for which translation-architect work is most urgently needed. They are also, by the analysis of the prior sections, the research programs for which the AI evaluation apparatus's trained preferences will most systematically suppress the relevant submissions.

**The Generalization Proof.** The researchers who will prove why overparameterized neural networks generalize will submit papers with citation profiles spanning statistical learning theory, algebraic geometry, and statistical physics. The reviewer matching system trained on ML venue citation data will match ML reviewers. The grant scoring system trained on funded ML proposals will find the algebraic geometry and statistical physics vocabulary anomalous. The literature discovery system will not have surfaced the algebraic geometry literature to these researchers in the first place, making the abductive recognition harder to reach before the paper is even conceived.

The locked room is not merely locked by human institutional resistance, as the prior documents established. It is now locked by an automated evaluation infrastructure that is actively trained against the key that would open it.

**Formal Technical Alignment.** A paper applying formal verification tools — temporal logic, abstract interpretation, model checking — to neural network behavioral constraints will have a citation profile spanning the formal methods community (CAV, LICS) and the ML community (NeurIPS, ICML). Current AI reviewer matching systems at ML conferences do not maintain reviewer pools with formal methods expertise. The papers these researchers need to cite — in *Formal Aspects of Computing*, in *ACM TOPLAS* — are not in the training citation graphs of ML conference reviewer-matching systems.

Sutskever's SSI is attempting the strategic dimension of this problem at scale. The technical dimension requires a synthesis the apparatus now has additional computational infrastructure to suppress.

**The Consciousness-Interpretability Convergence.** Anthropic's interpretability program is accumulating empirical findings at a remarkable rate. The foundational theoretical question — what distinguishes a system that processes information from one that represents it — is a question in philosophy of mind. Papers importing Friston's free energy formalism, Tononi's integrated information mathematics, or Dehaene's global workspace theory into interpretability research will have citation profiles spanning neuroscience and philosophy journals alongside ML publications.

Karpathy's presence at Anthropic places him adjacent to this synthesis. The prior documents establish that his training history has not positioned him to make it. The AI evaluation apparatus that will evaluate the papers of whoever might make it is now an additional structural barrier — one that was not present when Anthropic's interpretability program began and that deepens with each training generation of the review tools those papers will encounter.

The three locked rooms are not merely understaffed. They are now staffed by researchers whose papers face an evaluation apparatus specifically calibrated — by the machinery of correct learning from biased data — against the citation signatures and vocabulary anomalies that those papers necessarily produce.

---

## 9. The Identification Problem: You Cannot Calibrate from Filtered Data

The obvious prescription for the problem this document identifies is calibration: retrain the AI evaluation systems on a corpus that includes translation-architect papers at their correct base rate rather than the suppressed rate produced by the compound filter. Correct the distribution; correct the learned preference.

The prescription is correct. The implementation is structurally unavailable using the data the field currently has.

To calibrate a research evaluation AI against translation-architect work, ground truth labels are required: papers that are translation-architect papers, evaluated as such by competent observers, and shown to have citation trajectories consistent with genuine cross-paradigm synthesis — typically delayed high-impact rather than immediate high-citation. This ground truth corpus must be large enough to represent the distribution of translation-architect papers at a rate sufficient to influence the model's learned preferences.

That corpus does not exist. It does not exist because the compound filter has been removing translation-architect papers from the published record for decades. The papers that survived are the ones that passed a series of filters calibrated against them. The papers that did not survive — which by the compound filter analysis constitute the majority — are not in the published record. They are not available as negative examples for training (the filter removed them before publication) or as positive examples (the filter removed them before they could be identified and tracked as cross-paradigm successes).

The identification problem is structural: you cannot measure what the filter has been filtering out using data generated by the filter. Calibration requires a prior intervention — a deliberate, policy-driven effort to fund, accept, and longitudinally track translation-architect research — before the calibration data can be generated. But the AI evaluation apparatus is being deployed now, before that intervention has been decided, using training data that already reflects the filter's effects.

The apparatus is being trained before the ground truth is available. Its deployment will deepen the filter before the calibration data can be generated. The recursive degradation will make the calibration problem harder, not easier, with each deployment generation.

This is the sharpest version of the prior documents' institutional cascade analysis applied to the computational extension. The cascade does not merely persist; it is now automated at a rate faster than any institutional correction can follow — and the correction cannot be designed until the measurement is performed, and the measurement is performed on data the apparatus has already filtered.

The temporal ordering is not accidental. It is the structure of the problem.

---

## 10. Novel Theoretical Coverage

| Framework | Primary Sources | Core Contribution |
|-----------|----------------|-------------------|
| The Trained Horizon | Shumailov et al. (2024); this document | The paradigm's evaluative boundary, when encoded as an AI system's trained preference, becomes a computational horizon that is more consistent, more recursive, and less correctable than the institutional horizon it replaces; the horizon does not merely persist — it converges |
| Variance Protection and Its Elimination | Kahneman, Sibony & Sunstein (2021); this document | Human review variance, despite being a documented source of bias in paradigm-depth evaluation, is simultaneously the structural mechanism by which translation-architect papers occasionally survived the compound filter; AI consistency eliminates both forms of variance simultaneously, with no mechanism to distinguish the harmful variance from the protective variance |
| Model Collapse in Evaluation Systems | Shumailov et al. (2024); Merton (1968); this document | The Matthew Effect in citation networks — paradigm-internal signals compounding over time — is now being mechanized: AI evaluation systems trained on the accepted literature generate training distributions for their successors that progressively underrepresent the translation-architect tail, until the tail is absent from the model's learned preferences |
| Citation Homophily as Computational Preference | Merton (1968); Foster et al. (2015); this document | AI systems trained on citation graphs learn citation homophily as a positive quality signal; cross-paradigm citation anomalies — the bibliographic signature of inter-paradigm synthesis — are learned as negative quality signals without any instruction to this effect, as the correct representation of the training distribution |
| The Three Deployment Vectors | This document | Reviewer matching, grant scoring, and literature discovery constitute three distinct mechanisms by which AI evaluation encodes paradigm-depth preference at different stages of the research pipeline; the three vectors are not redundant — they amplify different stages of the compound filter simultaneously, including the pre-research literature discovery stage upstream of all prior stages |
| The Identification Problem in Calibration | This document | The AI evaluation apparatus cannot be calibrated against translation-architect work using historical publication data because the compound filter has systematically removed that work from the data before the apparatus was trained on it; calibration requires prior policy intervention, not merely better statistical technique; the temporal ordering forecloses the technical fix |
| The Compound Filter's Computational Extension | THE-SILO-WITHIN-THE-FRONTIER 1 (2026); this document | Stage 7 of the compound filter — AI evaluation — has three properties that make it qualitatively different from stages 1–6: variance suppression, recursive training, and scope expansion across all prior stages simultaneously; the three properties combine to produce a stage whose selection probability is strictly decreasing over training generations |
| The Upstream Suppression Effect | This document | The literature discovery vector suppresses the cross-paradigm abductive recognition before the research is designed, before the paper is written, before any review stage is reached; the trained horizon operates upstream of the compound filter's first stage, not merely within it |

---

## 11. Falsifiable Predictions

**P1 — The Citation Half-Life Divergence**

Translation-architect papers accepted to major ML conferences — a rare event for the reasons this document identifies — will show measurably longer citation half-lives than paradigm-depth papers accepted to the same conferences at the same time. The translation-architect paper's citations arrive slowly, from two paradigms that initially did not cite it, and compound as both paradigms absorb the synthesis. This trajectory is structurally distinct from the paradigm-depth paper's rapid-peak-then-decay pattern. *Falsification condition:* no statistically significant difference in citation half-life between papers classified as translation-architect (by cross-paradigm citation profile) and paradigm-depth papers in a 2018–2025 cohort, measured at 2030.

**P2 — The Reviewer Disagreement Signature**

Translation-architect papers submitted to major ML conferences will show a distinctive structured reviewer disagreement pattern: high variance across reviewers, with the disagreement aligned along paradigm lines — reviewers from the native paradigm scoring lower, reviewers with cross-paradigm exposure scoring higher. This is the variance-protection mechanism operating, and it is precisely what AI review assistance is eliminating. *Falsification condition:* no paradigm-structured reviewer disagreement on cross-paradigm submissions in an audit of OpenReview data from 2022–2025 conferences.

**P3 — The Generation Decay**

Among AI-assisted peer review tools deployed at major ML conferences between 2022 and 2028, successive versions will show measurably lower acceptance rates for papers with cross-paradigm citation profiles — defined as having >20% of citations outside the submitting venue's primary citation cluster — relative to papers with paradigm-internal citation profiles, controlling for overall acceptance rate. The decay will be monotonic across tool versions. *Falsification condition:* no decreasing trend in cross-paradigm paper acceptance rates at AI-assisted venues over the 2022–2028 period, measured against the 2018–2021 baseline.

**P4 — The Literature Discovery Anchoring**

Researchers who rely primarily on AI-powered literature discovery will show measurably narrower citation networks — lower inter-paradigm citation diversity — than researchers who use traditional search methods (direct database search, reference list traversal, librarian consultation) at the same career stage and in the same subfields. *Falsification condition:* no significant difference in citation network cross-paradigm diversity between AI-discovery and traditional-discovery researchers in a controlled study of 2024–2026 ML conference submissions.

**P5 — The Calibration Data Deficit**

Attempts to calibrate AI research evaluation systems against translation-architect work will be stymied by the identification problem this document describes: there will not be sufficient historical examples of high-quality translation-architect work — accepted, longitudinally tracked, confirmed to have produced cross-paradigm impact — at rates adequate for model calibration. *Falsification condition:* a corpus of 500+ translation-architect papers with confirmed cross-paradigm citation impact exists in the published record prior to 2024, sufficient to train a well-calibrated AI review model on cross-paradigm research quality.

**P6 — The Breakthrough Attribution Pattern**

The next major breakthrough in one of the three locked rooms — the generalization proof, formal technical alignment, or the consciousness-interpretability convergence — will have been submitted to and initially rejected by at least one major venue using AI-assisted reviewer matching. The initial rejection will be traceable to a reviewer-paper mismatch along paradigm lines: reviewers competent in one of the paper's paradigms but not the synthesis. This is the variance-protection mechanism being applied by residual human variance against the AI-consistency trend — the reviewer who was mismatched happened to have the expertise to recognize and advocate for the paper. *Falsification condition:* the breakthrough paper is accepted on first submission at its primary venue, with no documented reviewer mismatch along paradigm lines.

**P7 — The Training Data Audit**

No major AI-assisted research evaluation tool currently deployed at ML conferences, funding agencies, or literature discovery platforms will, on systematic audit, show a training corpus containing translation-architect papers — defined by cross-paradigm citation profiles — at rates exceeding 2% of total training examples. The tools are trained on what was accepted; what was accepted reflects the prior compound filter. *Falsification condition:* at least one deployed evaluation tool demonstrates >5% representation of cross-paradigm papers in its training corpus, with documentation.

---

## 12. Complete Evidence Base

| Source | Citation | Core Finding Applied |
|--------|----------|---------------------|
| Shumailov, I., et al. (2024) | "AI models collapse when trained on recursively self-generated data." *Nature*, 631, 755–759 | Model collapse as the formal analog: recursive training of evaluation systems on filtered outputs produces progressive loss of the translation-architect tail; the mechanism is identical in structure to the evaluation apparatus's recursive degradation |
| Merton, R.K. (1968) | "The Matthew Effect in Science." *Science*, 159(3810), 56–63 | The foundational paradigm-internal amplification mechanism: prior recognition compounds future recognition; now being mechanized through AI training on citation networks that encode this compounding |
| Kahneman, D., Sibony, O., & Sunstein, C.R. (2021) | *Noise: A Flaw in Human Judgment*. Little, Brown Spark | Variance reduction as improvement only when the mean is unbiased; the case against automated consistency in evaluation systems whose training distribution is paradigm-depth-biased |
| Liang, W., et al. (2024) | "Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews." *arXiv* | Direct evidence of LLM penetration into ML peer review at scale; the empirical baseline for the extent of AI involvement in the evaluation apparatus this document analyzes |
| Foster, J.G., & Evans, J.A. (2015) | "Tradition and innovation in scientists' research strategies." *American Sociological Review*, 80(5), 875–908 | Conventional research strategies receive more citations in the short term; unconventional strategies — including cross-paradigm synthesis — show higher long-term impact; the citation signal AI evaluation systems learn from is the short-term paradigm-depth signal, not the long-term translation-architect signal |
| Tomkins, A., et al. (2017) | "Reviewer bias in single- versus double-blind peer review." *PNAS*, 114(48), 12708–12713 | Established baseline on structured reviewer bias; the variance structure of human review whose elimination by AI consistency this document analyzes |
| Peterson, A.J. (2025) | *arXiv:2508.19625*, University of Poitiers | Monotonic over-investment distortion under AI prevalence; applied here to the evaluation apparatus, not only to the individual researcher's skill formation |
| Kapoor, S., et al. (2026) | *arXiv:2605.20520* | Open-world evaluation as the human-premium zone; the three locked rooms as the specific problem classes requiring translation-architect work that the AI apparatus is being trained against |
| Stanford HAI (2026) | *2026 AI Index Report* | Benchmark saturation across paradigm-depth ML domains; the increasing value of the frontier the trained apparatus is suppressing entry to, measured contemporaneously with the apparatus's deployment |
| Collins, H., & Evans, R. (2002) | *Social Studies of Science*, 32(2), 235–296 | Contributory expertise as the threshold for genuine cross-paradigm evaluation; why reviewer matching systems cannot provide competent review of translation-architect work even when they attempt to |
| Kuhn, T.S. (1962) | *The Structure of Scientific Revolutions*. University of Chicago Press | Paradigm as the epistemological framework whose horizon is being encoded; normal vs. revolutionary science as the distinction the apparatus is trained to enforce |
| Boden, M.A. (2004) | *The Creative Mind* (2nd ed.). Routledge | Transformational creativity at the paradigm level; the type of work the apparatus is learning to score as anomalous by correct inference from its training distribution |
| Peirce, C.S. (1887) | "A guess at the riddle" | Abduction as the only logic that introduces new ideas; the operation the apparatus is trained to recognize as citation-anomalous and therefore low-quality |
| Acemoglu, D., & Restrepo, P. (2018) | *American Economic Review*, 108(6) | Displacement vs. reinstatement; the AI evaluation apparatus as a displacement mechanism that closes the reinstatement channel for translation-architect researcher entry before the channel is measured |
| Restuccia, D., & Rogerson, R. (2017) | *Journal of Economic Perspectives*, 31(3) | Factor misallocation as the primary TFP driver; the AI apparatus compounds the cognitive capital misallocation by making entry to high-paradigm-translation research positions computationally harder at each training generation |
| Ballantyne, N. (2019) | "Epistemic trespassing." *Mind*, 128(510), 367–395 | Why paradigm-internal reviewers treat cross-paradigm synthesis as epistemically suspect; this individual disposition is now being encoded as the mean trained preference of the AI evaluation apparatus |
| Sunstein, C.R., & Thaler, R. (2008) | *Nudge*. Yale University Press | Informational cascade; the deployment of AI evaluation tools produces a cascade in which each institution trusts the prior institution's tool selection, and no institution audits the training distribution the tool was built on |
| Pluess, M., & Belsky, J. (2013) | *Psychological Bulletin*, 139(4), 901–916 | Vantage sensitivity; the translation-architect profile develops under enriched environments and is the profile most suppressed by the systematic environmental depletion this document's evaluation apparatus analysis describes |

---

## Conclusion: The Horizon Has Learned to Stay

The prior documents in this lineage established a precise diagnosis: the prestige apparatus is miscalibrated, the compound filter is multiplicative, and the researchers most needed for the field's hardest problems are systematically under-identified and underfunded. The prescription was institutional: build new evaluation instruments, restructure doctoral pipelines, found inter-paradigm labs.

This document has established that the prescription now faces a constraint the prior analysis did not anticipate: the institutions are automating their evaluation functions, and the automation is being trained on the biased outputs those institutions have already produced.

The horizon — the paradigm's edge, beyond which the apparatus cannot see — was always there. But horizons are not fixed. Human institutions, for all their inertia, had variance. A particularly insightful program chair, a particularly cross-trained reviewer, a particularly imaginative grant officer — the institutional apparatus occasionally produced the conditions under which a translation-architect paper could be recognized. The probability was low. It was not zero.

The AI apparatus is eliminating the nonzero term. It is not doing so through any failure of engineering. It is doing so by being an excellent learner of the distribution it was given. The distribution was generated by the biased filter. The learner is excellent. The outcome follows.

The Litt discomfort — the specific discomfort Daniel Litt expressed upon realizing that the Erdős problem sat unsolved for seventy years while the tools for its solution existed in a different room — is the right discomfort here too. But in this document's version of the Litt moment, the different room is not merely unlabeled and unvisited. The apparatus being built to direct researchers toward relevant literature is learning not to surface it. The apparatus being built to match reviewers with papers is learning not to match the competent ones. The apparatus being built to identify fundable proposals is learning not to fund the right ones.

And it is learning all of this from data. Correctly. In accordance with its training objective. With each generation, better.

The field should share the Litt discomfort. Then it should do something harder than any prior iteration of the discomfort demanded: audit the training data of the evaluation apparatus it is currently deploying — before that data defines, permanently and recursively, the horizon the apparatus will enforce on the next generation of researchers who might otherwise have crossed it.

The intervention must precede the deployment. The deployment is already underway.

---

*Evidence base current as of May 2026. All predictions are stated in falsifiable form with explicit conditions. Researchers who contest the model collapse analog or the compound filter's computational extension are invited to specify the training data distributions of the AI evaluation systems they use and to demonstrate that translation-architect papers are represented in those distributions at rates sufficient to influence the models' learned preferences.*

*Prior analyses in this lineage: THE-SILOED-MIND · THE-SILO-WITHIN-THE-FRONTIER · CHRONOLOGICAL-TYRANNY · COMPETITIVE-EVACUATION · THE-COGNITIVE-TAX · THE-CRYPTIC-PHENOTYPE · THE-RATIONAL-STOP · THE-COMPLETION-COMPULSION · THE-REINSTATEMENT-PREMIUM · CAPACITY-SUPPRESSION · PROXIMITY-AS-WEAPON · NARRATIVE-DISPOSSESSION · THE-EXIT-THEOREM*

*ERI Labs · Eric Ren · Jersey City, New Jersey · github.com/ericrenone · May 2026*
