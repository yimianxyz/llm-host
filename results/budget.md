# Budget log ($200 cap)

| Time (UTC) | Event | Balance read | Pod-hours so far | Note |
|---|---|---|---|---|
| 2026-11-08T19:16Z | before any pod | $200.00 (runpodctl user) | 0 | spend limit $80/hr |
| 2026-11-08T19:18Z | rehearsal pod 0ahvagkjn8k4pw created (1x A40, CA-MTL-1) | - | - | $0.59/hr |
| 2026-11-08T19:19Z | main pod fbsy2njw0xt5mp created (4x H200 SXM, EUR-IS-5) | - | - | $21.16/hr; deadline 9.1 h |
| 2026-11-08T20:36Z | Step 1 done (B4 serving) | $173.226279286 (runpodctl user) | ~1.3 | config debugging B0-B3 cost ~35 min |
| 2026-11-08T21:30Z | Step 3 done (baseline + reference shapes) | $153.7645384526 (runpodctl user) | ~2.2 | c=16 stopped early (collapse) |
| 2026-11-08T22:15Z | L3 kept, L5p reverted, L2 running | $137.8303456525 (runpodctl user) | ~2.9 | |
| 2026-11-08T23:33Z | L7 DSpark reverted; L4 at 1M did not boot; L5c+L4c running | $111.2998311639 (runpodctl user) | ~4.2 | about 5.0 h of balance left at $21.16/hr; deadline ~04:25Z |
| 2026-11-09T00:19Z | Gate B: L5d best; Step 6 running | $93.5767632972 (runpodctl user) | ~5.0 | deadline ~04:26Z |
