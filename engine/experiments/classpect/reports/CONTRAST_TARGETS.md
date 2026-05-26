# Classpect Contrast Targets

These targets come from DSPy grouped-CV diagnostics. Use them to guide anonymous description generation without leaking class/aspect labels into generated descriptions.

## Purpose

The v2 description dataset should emphasize contrastive personality evidence for recurring mistakes, not broader or longer character summaries.

Generated descriptions must remain anonymous and must not include names, aliases, handles, initials, typing quirks, species, blood color, caste, gender, dream moon, lusus, planet, weapons, powers, canonical title, class, aspect, or direct plot identifiers.

## Class Confusions To Sharpen

| Repeated Mistake | Description Evidence To Emphasize |
| --- | --- |
| `Heir -> Knight` | Distinguish passive adaptation, being carried by circumstances, and diffuse influence from defensive duty, guarded competence, and protective service. |
| `Seer -> Mage` | Distinguish interpretive guidance and advising others from personally embodied, hard-earned understanding. |
| `Thief -> Prince` | Distinguish acquisitive self-assertion and taking agency/resources from destructive rejection, negation, or tearing down. |
| `Maid -> Mage` | Distinguish self-creation through service/maintenance from analysis born from personal suffering or expertise. |
| `Page -> Knight` | Distinguish latent potential, insecurity, and slow growth from already-practiced defense, service, and competence under pressure. |
| `Rogue -> Knight` | Distinguish redistributing, sharing, or indirect support from direct protective duty and self-armoring. |

## Aspect Confusions To Sharpen

| Repeated Mistake | Description Evidence To Emphasize |
| --- | --- |
| `Light -> Mind` | Distinguish attention to knowledge, relevance, luck, and visibility from choice logic, judgment, consequences, and decision trees. |
| `Breath -> Hope` | Distinguish independence, detachment, motion, and freedom from conviction, faith, optimism, and belief-maintenance. |
| `Space -> Life` | Distinguish patience, placement, environment, creation, and scale from growth, vitality, caretaking, appetite, and recovery. |
| `Space -> Light` | Distinguish spatial/contextual awareness and creation from relevance-seeking, knowledge curation, and being seen. |
| `Life -> Hope` | Distinguish nurturing, abundance, recovery, and vitality from idealism, faith, and belief-driven persistence. |
| `Doom -> Mind` | Distinguish limits, costs, fatalism, obligation, and consequences accepted as constraints from abstract choice analysis and judgment. |

## Prediction Biases To Watch

Recent local model direct/no-think runs over-predicted `Knight`, `Prince`, `Mage`, `Hope`, and `Mind`, while under-predicting labels such as `Heir`, `Space`, `Time`, `Breath`, and `Void`.

The v2 dataset should not force balance artificially, but descriptions should include enough evidence for under-predicted labels to be recoverable when appropriate.
